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
def _reset_triggers(self) -> None:
    """Set all trigger values in our state dictionary to False."""
    for state_id in self._new_widget_state:
        metadata = self._new_widget_state.widget_metadata.get(state_id)
        if metadata is not None:
            if metadata.value_type == 'trigger_value':
                self._new_widget_state[state_id] = Value(False)
            elif metadata.value_type == 'string_trigger_value':
                self._new_widget_state[state_id] = Value(None)
    for state_id in self._old_state:
        metadata = self._new_widget_state.widget_metadata.get(state_id)
        if metadata is not None:
            if metadata.value_type == 'trigger_value':
                self._old_state[state_id] = False
            elif metadata.value_type == 'string_trigger_value':
                self._old_state[state_id] = None