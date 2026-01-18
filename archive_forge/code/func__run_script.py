from __future__ import annotations
import gc
import sys
import threading
import types
from contextlib import contextmanager
from enum import Enum
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Callable, Final
from blinker import Signal
from streamlit import config, runtime, source_util, util
from streamlit.error_util import handle_uncaught_app_exception
from streamlit.logger import get_logger
from streamlit.proto.ClientState_pb2 import ClientState
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.scriptrunner.script_requests import (
from streamlit.runtime.scriptrunner.script_run_context import (
from streamlit.runtime.state import (
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.vendor.ipython.modified_sys_path import modified_sys_path
def _run_script(self, rerun_data: RerunData) -> None:
    """Run our script.

        Parameters
        ----------
        rerun_data: RerunData
            The RerunData to use.

        """
    from streamlit.delta_generator import dg_stack
    assert self._is_in_script_thread()
    while True:
        _LOGGER.debug('Running script %s', rerun_data)
        start_time: float = timer()
        prep_time: float = 0
        runtime.get_instance().media_file_mgr.clear_session_refs()
        main_script_path = self._main_script_path
        pages = source_util.get_pages(main_script_path)
        main_page_info = list(pages.values())[0]
        current_page_info = None
        uncaught_exception = None
        if rerun_data.page_script_hash:
            current_page_info = pages.get(rerun_data.page_script_hash, None)
        elif not rerun_data.page_script_hash and rerun_data.page_name:
            current_page_info = next(filter(lambda p: p and p['page_name'] == rerun_data.page_name, pages.values()), None)
        else:
            current_page_info = main_page_info
        page_script_hash = current_page_info['page_script_hash'] if current_page_info is not None else main_page_info['page_script_hash']
        fragment_ids_this_run = set(rerun_data.fragment_id_queue)
        ctx = self._get_script_run_ctx()
        ctx.reset(query_string=rerun_data.query_string, page_script_hash=page_script_hash, fragment_ids_this_run=fragment_ids_this_run)
        self.on_event.send(self, event=ScriptRunnerEvent.SCRIPT_STARTED, page_script_hash=page_script_hash, fragment_ids_this_run=fragment_ids_this_run)
        try:
            if current_page_info:
                script_path = current_page_info['script_path']
            else:
                script_path = main_script_path
                msg = ForwardMsg()
                msg.page_not_found.page_name = rerun_data.page_name
                ctx.enqueue(msg)
            code = self._script_cache.get_bytecode(script_path)
        except Exception as ex:
            _LOGGER.debug('Fatal script error: %s', ex)
            self._session_state[SCRIPT_RUN_WITHOUT_ERRORS_KEY] = False
            self.on_event.send(self, event=ScriptRunnerEvent.SCRIPT_STOPPED_WITH_COMPILE_ERROR, exception=ex)
            return
        if config.get_option('runner.installTracer'):
            self._install_tracer()
        rerun_exception_data: RerunData | None = None
        original_cursors = ctx.cursors
        original_dg_stack = dg_stack.get()
        premature_stop: bool = False
        try:
            module = self._new_module('__main__')
            sys.modules['__main__'] = module
            module.__dict__['__file__'] = script_path
            with modified_sys_path(self._main_script_path), self._set_execing_flag():
                if rerun_data.widget_states is not None:
                    self._session_state.on_script_will_rerun(rerun_data.widget_states)
                ctx.on_script_start()
                prep_time = timer() - start_time
                if rerun_data.fragment_id_queue:
                    for fragment_id in rerun_data.fragment_id_queue:
                        try:
                            wrapped_fragment = self._fragment_storage.get(fragment_id)
                            ctx.current_fragment_id = fragment_id
                            wrapped_fragment()
                        except KeyError:
                            raise RuntimeError(f'Could not find fragment with id {fragment_id}')
                else:
                    self._fragment_storage.clear()
                    exec(code, module.__dict__)
                self._session_state.maybe_check_serializable()
                self._session_state[SCRIPT_RUN_WITHOUT_ERRORS_KEY] = True
        except RerunException as e:
            rerun_exception_data = e.rerun_data
            ctx.cursors = original_cursors
            dg_stack.set(original_dg_stack)
            premature_stop = False
        except StopException:
            premature_stop = True
        except Exception as ex:
            self._session_state[SCRIPT_RUN_WITHOUT_ERRORS_KEY] = False
            uncaught_exception = ex
            handle_uncaught_app_exception(uncaught_exception)
            premature_stop = True
        finally:
            if rerun_exception_data:
                finished_event = ScriptRunnerEvent.SCRIPT_STOPPED_FOR_RERUN
            elif rerun_data.fragment_id_queue:
                finished_event = ScriptRunnerEvent.FRAGMENT_STOPPED_WITH_SUCCESS
            else:
                finished_event = ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS
            if ctx.gather_usage_stats:
                try:
                    from streamlit.runtime.metrics_util import create_page_profile_message, to_microseconds
                    ctx.enqueue(create_page_profile_message(ctx.tracked_commands, exec_time=to_microseconds(timer() - start_time), prep_time=to_microseconds(prep_time), uncaught_exception=type(uncaught_exception).__name__ if uncaught_exception else None))
                except Exception as ex:
                    _LOGGER.debug('Failed to create page profile', exc_info=ex)
            self._on_script_finished(ctx, finished_event, premature_stop)
        _log_if_error(_clean_problem_modules)
        if rerun_exception_data is not None:
            rerun_data = rerun_exception_data
        else:
            break