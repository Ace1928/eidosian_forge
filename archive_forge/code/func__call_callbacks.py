from __future__ import annotations
import json
import pickle
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import (
from typing_extensions import TypeAlias
import streamlit as st
from streamlit import config, util
from streamlit.errors import StreamlitAPIException, UnserializableSessionStateError
from streamlit.proto.WidgetStates_pb2 import WidgetState as WidgetStateProto
from streamlit.proto.WidgetStates_pb2 import WidgetStates as WidgetStatesProto
from streamlit.runtime.state.common import (
from streamlit.runtime.state.query_params import QueryParams
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.type_util import ValueFieldName, is_array_value_field_name
def _call_callbacks(self) -> None:
    """Call any callback associated with each widget whose value
        changed between the previous and current script runs.
        """
    from streamlit.runtime.scriptrunner import RerunException
    changed_widget_ids = [wid for wid in self._new_widget_state if self._widget_changed(wid)]
    for wid in changed_widget_ids:
        try:
            self._new_widget_state.call_callback(wid)
        except RerunException:
            st.warning('Calling st.rerun() within a callback is a no-op.')