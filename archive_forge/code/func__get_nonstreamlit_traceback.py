from __future__ import annotations
import os
import traceback
from typing import TYPE_CHECKING, Final, cast
import streamlit
from streamlit.errors import (
from streamlit.logger import get_logger
from streamlit.proto.Exception_pb2 import Exception as ExceptionProto
from streamlit.runtime.metrics_util import gather_metrics
def _get_nonstreamlit_traceback(extracted_tb: traceback.StackSummary) -> list[traceback.FrameSummary]:
    return [entry for entry in extracted_tb if not _is_in_streamlit_package(entry.filename)]