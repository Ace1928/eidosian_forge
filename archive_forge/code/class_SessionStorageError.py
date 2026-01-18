from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol, cast
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
class SessionStorageError(Exception):
    """Exception class for errors raised by SessionStorage.

    The original error that causes a SessionStorageError to be (re)raised will generally
    be an I/O error specific to the concrete SessionStorage implementation.
    """