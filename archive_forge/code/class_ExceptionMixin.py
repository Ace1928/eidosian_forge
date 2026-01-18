from __future__ import annotations
import os
import traceback
from typing import TYPE_CHECKING, Final, cast
import streamlit
from streamlit.errors import (
from streamlit.logger import get_logger
from streamlit.proto.Exception_pb2 import Exception as ExceptionProto
from streamlit.runtime.metrics_util import gather_metrics
class ExceptionMixin:

    @gather_metrics('exception')
    def exception(self, exception: BaseException) -> DeltaGenerator:
        """Display an exception.

        Parameters
        ----------
        exception : Exception
            The exception to display.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> e = RuntimeError('This is an exception of type RuntimeError')
        >>> st.exception(e)

        """
        exception_proto = ExceptionProto()
        marshall(exception_proto, exception)
        return self.dg._enqueue('exception', exception_proto)

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)