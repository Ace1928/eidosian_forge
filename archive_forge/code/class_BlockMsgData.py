from __future__ import annotations
import contextlib
import hashlib
import threading
import types
from dataclasses import dataclass
from typing import (
from google.protobuf.message import Message
import streamlit as st
from streamlit import runtime, util
from streamlit.elements import NONWIDGET_ELEMENTS, WIDGETS
from streamlit.logger import get_logger
from streamlit.proto.Block_pb2 import Block
from streamlit.runtime.caching.cache_errors import (
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.hashing import update_hash
from streamlit.runtime.scriptrunner.script_run_context import (
from streamlit.runtime.state.common import WidgetMetadata
from streamlit.util import HASHLIB_KWARGS
@dataclass(frozen=True)
class BlockMsgData:
    message: Block
    id_of_dg_called_on: str
    returned_dgs_id: str