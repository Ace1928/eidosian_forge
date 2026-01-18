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
def _on_scriptrunner_event(self, sender: ScriptRunner | None, event: ScriptRunnerEvent, forward_msg: ForwardMsg | None=None, exception: BaseException | None=None, client_state: ClientState | None=None, page_script_hash: str | None=None, fragment_ids_this_run: set[str] | None=None) -> None:
    """Called when our ScriptRunner emits an event.

        This is generally called from the sender ScriptRunner's script thread.
        We forward the event on to _handle_scriptrunner_event_on_event_loop,
        which will be called on the main thread.
        """
    self._event_loop.call_soon_threadsafe(lambda: self._handle_scriptrunner_event_on_event_loop(sender, event, forward_msg, exception, client_state, page_script_hash, fragment_ids_this_run))