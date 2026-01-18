from __future__ import annotations
import os
import traceback
from typing import TYPE_CHECKING, Final, cast
import streamlit
from streamlit.errors import (
from streamlit.logger import get_logger
from streamlit.proto.Exception_pb2 import Exception as ExceptionProto
from streamlit.runtime.metrics_util import gather_metrics
def _is_in_streamlit_package(file: str) -> bool:
    """True if the given file is part of the streamlit package."""
    try:
        common_prefix = os.path.commonprefix([os.path.realpath(file), _STREAMLIT_DIR])
    except ValueError:
        return False
    return common_prefix == _STREAMLIT_DIR