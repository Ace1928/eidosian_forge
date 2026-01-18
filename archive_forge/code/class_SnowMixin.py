from __future__ import annotations
from typing import TYPE_CHECKING, cast
from streamlit.proto.Snow_pb2 import Snow as SnowProto
from streamlit.runtime.metrics_util import gather_metrics
class SnowMixin:

    @gather_metrics('snow')
    def snow(self) -> DeltaGenerator:
        """Draw celebratory snowfall.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> st.snow()

        ...then watch your app and get ready for a cool celebration!

        """
        snow_proto = SnowProto()
        snow_proto.show = True
        return self.dg._enqueue('snow', snow_proto)

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)