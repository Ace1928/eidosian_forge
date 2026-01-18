from __future__ import annotations
import asyncio
import sys
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Callable, Final
import streamlit.elements.exception as exception_utils
from streamlit import config, runtime, source_util
from streamlit.case_converters import to_snake_case
from streamlit.logger import get_logger
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ClientState_pb2 import ClientState
from streamlit.proto.Common_pb2 import FileURLs, FileURLsRequest
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.GitInfo_pb2 import GitInfo
from streamlit.proto.NewSession_pb2 import (
from streamlit.proto.PagesChanged_pb2 import PagesChanged
from streamlit.runtime import caching, legacy_caching
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.fragment import FragmentStorage, MemoryFragmentStorage
from streamlit.runtime.metrics_util import Installation
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner import RerunData, ScriptRunner, ScriptRunnerEvent
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.secrets import secrets_singleton
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.version import STREAMLIT_VERSION_STRING
from streamlit.watcher import LocalSourcesWatcher
def _handle_scriptrunner_event_on_event_loop(self, sender: ScriptRunner | None, event: ScriptRunnerEvent, forward_msg: ForwardMsg | None=None, exception: BaseException | None=None, client_state: ClientState | None=None, page_script_hash: str | None=None, fragment_ids_this_run: set[str] | None=None) -> None:
    """Handle a ScriptRunner event.

        This function must only be called on our eventloop thread.

        Parameters
        ----------
        sender : ScriptRunner | None
            The ScriptRunner that emitted the event. (This may be set to
            None when called from `handle_backmsg_exception`, if no
            ScriptRunner was active when the backmsg exception was raised.)

        event : ScriptRunnerEvent
            The event type.

        forward_msg : ForwardMsg | None
            The ForwardMsg to send to the frontend. Set only for the
            ENQUEUE_FORWARD_MSG event.

        exception : BaseException | None
            An exception thrown during compilation. Set only for the
            SCRIPT_STOPPED_WITH_COMPILE_ERROR event.

        client_state : streamlit.proto.ClientState_pb2.ClientState | None
            The ScriptRunner's final ClientState. Set only for the
            SHUTDOWN event.

        page_script_hash : str | None
            A hash of the script path corresponding to the page currently being
            run. Set only for the SCRIPT_STARTED event.

        fragment_ids_this_run : set[str] | None
            The fragment IDs of the fragments being executed in this script run. Only
            set for the SCRIPT_STARTED event. If this value is falsy, this script run
            must be for the full script.
        """
    assert self._event_loop == asyncio.get_running_loop(), 'This function must only be called on the eventloop thread the AppSession was created on.'
    if sender is not self._scriptrunner:
        _LOGGER.debug('Ignoring event from non-current ScriptRunner: %s', event)
        return
    prev_state = self._state
    if event == ScriptRunnerEvent.SCRIPT_STARTED:
        if self._state != AppSessionState.SHUTDOWN_REQUESTED:
            self._state = AppSessionState.APP_IS_RUNNING
        assert page_script_hash is not None, 'page_script_hash must be set for the SCRIPT_STARTED event'
        if not fragment_ids_this_run:
            self._clear_queue()
        self._enqueue_forward_msg(self._create_new_session_message(page_script_hash, fragment_ids_this_run))
    elif event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS or event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_COMPILE_ERROR or event == ScriptRunnerEvent.FRAGMENT_STOPPED_WITH_SUCCESS:
        if self._state != AppSessionState.SHUTDOWN_REQUESTED:
            self._state = AppSessionState.APP_NOT_RUNNING
        if event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS:
            status = ForwardMsg.FINISHED_SUCCESSFULLY
        elif event == ScriptRunnerEvent.FRAGMENT_STOPPED_WITH_SUCCESS:
            status = ForwardMsg.FINISHED_FRAGMENT_RUN_SUCCESSFULLY
        else:
            status = ForwardMsg.FINISHED_WITH_COMPILE_ERROR
        self._enqueue_forward_msg(self._create_script_finished_message(status))
        self._debug_last_backmsg_id = None
        if event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS or event == ScriptRunnerEvent.FRAGMENT_STOPPED_WITH_SUCCESS:
            if self._local_sources_watcher:
                self._local_sources_watcher.update_watched_modules()
        else:
            assert exception is not None, 'exception must be set for the SCRIPT_STOPPED_WITH_COMPILE_ERROR event'
            msg = ForwardMsg()
            exception_utils.marshall(msg.session_event.script_compilation_exception, exception)
            self._enqueue_forward_msg(msg)
    elif event == ScriptRunnerEvent.SCRIPT_STOPPED_FOR_RERUN:
        self._state = AppSessionState.APP_NOT_RUNNING
        self._enqueue_forward_msg(self._create_script_finished_message(ForwardMsg.FINISHED_EARLY_FOR_RERUN))
        if self._local_sources_watcher:
            self._local_sources_watcher.update_watched_modules()
    elif event == ScriptRunnerEvent.SHUTDOWN:
        assert client_state is not None, 'client_state must be set for the SHUTDOWN event'
        if self._state == AppSessionState.SHUTDOWN_REQUESTED:
            runtime.get_instance().media_file_mgr.clear_session_refs(self.id)
        self._client_state = client_state
        self._scriptrunner = None
    elif event == ScriptRunnerEvent.ENQUEUE_FORWARD_MSG:
        assert forward_msg is not None, 'null forward_msg in ENQUEUE_FORWARD_MSG event'
        self._enqueue_forward_msg(forward_msg)
    app_was_running = prev_state == AppSessionState.APP_IS_RUNNING
    app_is_running = self._state == AppSessionState.APP_IS_RUNNING
    if app_is_running != app_was_running:
        self._enqueue_forward_msg(self._create_session_status_changed_message())