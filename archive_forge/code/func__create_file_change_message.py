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
def _create_file_change_message(self) -> ForwardMsg:
    """Create and return a 'script_changed_on_disk' ForwardMsg."""
    msg = ForwardMsg()
    msg.session_event.script_changed_on_disk = True
    return msg