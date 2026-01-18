from __future__ import annotations
import hashlib
import inspect
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence
from unittest.mock import MagicMock
from urllib import parse
from streamlit import source_util
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime import Runtime
from streamlit.runtime.caching.storage.dummy_cache_storage import (
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.runtime.secrets import Secrets
from streamlit.runtime.state.common import TESTING_KEY
from streamlit.runtime.state.safe_session_state import SafeSessionState
from streamlit.runtime.state.session_state import SessionState
from streamlit.testing.v1.element_tree import (
from streamlit.testing.v1.local_script_runner import LocalScriptRunner
from streamlit.testing.v1.util import patch_config_options
from streamlit.util import HASHLIB_KWARGS, calc_md5
Get elements or widgets of the specified type.

        This method returns the collection of all elements or widgets of
        the specified type on the current page. Retrieve a specific element by
        using its index (order on page) or key lookup.

        Parameters
        ----------
        element_type: str
            An element attribute of ``AppTest``. For example, "button",
            "caption", or "chat_input".

        Returns
        -------
        Sequence of Elements
            Sequence of elements of the given type. Individual elements can
            be accessed from a Sequence by index (order on the page). When
            getting and ``element_type`` that is a widget, individual widgets
            can be accessed by key. For example, ``at.get("text")[0]`` for the
            first ``st.text`` element or ``at.get("slider")(key="my_key")`` for
            the ``st.slider`` widget with a given key.
        