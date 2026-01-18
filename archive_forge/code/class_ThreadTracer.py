from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import (get_current_thread_id, NO_FTRACE,
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from _pydevd_bundle.pydevd_frame import PyDBFrame, is_unhandled_exception
class ThreadTracer(object):

    def __init__(self, args):
        self._args = args

    def __call__(self, frame, event, arg):
        """ This is the callback used when we enter some context in the debugger.

        We also decorate the thread we are in with info about the debugging.
        The attributes added are:
            pydev_state
            pydev_step_stop
            pydev_step_cmd
            pydev_notify_kill

        :param PyDB py_db:
            This is the global debugger (this method should actually be added as a method to it).
        """
        py_db, t, additional_info, cache_skips, frame_skips_cache = self._args
        if additional_info.is_tracing:
            return None if event == 'call' else NO_FTRACE
        additional_info.is_tracing += 1
        try:
            pydev_step_cmd = additional_info.pydev_step_cmd
            is_stepping = pydev_step_cmd != -1
            if py_db.pydb_disposed:
                return None if event == 'call' else NO_FTRACE
            if not is_thread_alive(t):
                py_db.notify_thread_not_alive(get_current_thread_id(t))
                return None if event == 'call' else NO_FTRACE
            frame_cache_key = frame.f_code
            if frame_cache_key in cache_skips:
                if not is_stepping:
                    return None if event == 'call' else NO_FTRACE
                elif cache_skips.get(frame_cache_key) == 1:
                    if additional_info.pydev_original_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE) and (not _global_notify_skipped_step_in):
                        notify_skipped_step_in_because_of_filters(py_db, frame)
                    back_frame = frame.f_back
                    if back_frame is not None and pydev_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE):
                        back_frame_cache_key = back_frame.f_code
                        if cache_skips.get(back_frame_cache_key) == 1:
                            return None if event == 'call' else NO_FTRACE
                    else:
                        return None if event == 'call' else NO_FTRACE
            try:
                abs_path_canonical_path_and_base = NORM_PATHS_AND_BASE_CONTAINER[frame.f_code.co_filename]
            except:
                abs_path_canonical_path_and_base = get_abs_path_real_path_and_base_from_frame(frame)
            file_type = py_db.get_file_type(frame, abs_path_canonical_path_and_base)
            if file_type is not None:
                if file_type == 1:
                    if not py_db.in_project_scope(frame, abs_path_canonical_path_and_base[0]):
                        cache_skips[frame_cache_key] = 1
                        return None if event == 'call' else NO_FTRACE
                else:
                    cache_skips[frame_cache_key] = 1
                    return None if event == 'call' else NO_FTRACE
            if py_db.is_files_filter_enabled:
                if py_db.apply_files_filter(frame, abs_path_canonical_path_and_base[0], False):
                    cache_skips[frame_cache_key] = 1
                    if is_stepping and additional_info.pydev_original_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE) and (not _global_notify_skipped_step_in):
                        notify_skipped_step_in_because_of_filters(py_db, frame)
                    back_frame = frame.f_back
                    if back_frame is not None and pydev_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE):
                        if py_db.apply_files_filter(back_frame, back_frame.f_code.co_filename, False):
                            back_frame_cache_key = back_frame.f_code
                            cache_skips[back_frame_cache_key] = 1
                            return None if event == 'call' else NO_FTRACE
                    else:
                        return None if event == 'call' else NO_FTRACE
            ret = PyDBFrame((py_db, abs_path_canonical_path_and_base, additional_info, t, frame_skips_cache, frame_cache_key)).trace_dispatch(frame, event, arg)
            if ret is None:
                cache_skips[frame_cache_key] = 2
                return None if event == 'call' else NO_FTRACE
            frame.f_trace = ret
            return ret
        except SystemExit:
            return None if event == 'call' else NO_FTRACE
        except Exception:
            if py_db.pydb_disposed:
                return None if event == 'call' else NO_FTRACE
            try:
                if pydev_log_exception is not None:
                    pydev_log_exception()
            except:
                pass
            return None if event == 'call' else NO_FTRACE
        finally:
            additional_info.is_tracing -= 1