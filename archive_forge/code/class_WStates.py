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
@dataclass
class WStates(MutableMapping[str, Any]):
    """A mapping of widget IDs to values. Widget values can be stored in
    serialized or deserialized form, but when values are retrieved from the
    mapping, they'll always be deserialized.
    """
    states: dict[str, WState] = field(default_factory=dict)
    widget_metadata: dict[str, WidgetMetadata[Any]] = field(default_factory=dict)

    def __repr__(self):
        return util.repr_(self)

    def __getitem__(self, k: str) -> Any:
        """Return the value of the widget with the given key.
        If the widget's value is currently stored in serialized form, it
        will be deserialized first.
        """
        wstate = self.states.get(k)
        if wstate is None:
            raise KeyError(k)
        if isinstance(wstate, Value):
            return wstate.value
        metadata = self.widget_metadata.get(k)
        if metadata is None:
            raise KeyError(k)
        value_field_name = cast(ValueFieldName, wstate.value.WhichOneof('value'))
        value = wstate.value.__getattribute__(value_field_name) if value_field_name else None
        if is_array_value_field_name(value_field_name):
            value = value.data
        elif value_field_name == 'json_value':
            value = json.loads(value)
        deserialized = metadata.deserializer(value, metadata.id)
        self.set_widget_metadata(replace(metadata, value_type=value_field_name))
        self.states[k] = Value(deserialized)
        return deserialized

    def __setitem__(self, k: str, v: WState) -> None:
        self.states[k] = v

    def __delitem__(self, k: str) -> None:
        del self.states[k]

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self):
        yield from self.states

    def keys(self) -> KeysView[str]:
        return KeysView(self.states)

    def items(self) -> set[tuple[str, Any]]:
        return {(k, self[k]) for k in self}

    def values(self) -> set[Any]:
        return {self[wid] for wid in self}

    def update(self, other: WStates) -> None:
        """Copy all widget values and metadata from 'other' into this mapping,
        overwriting any data in this mapping that's also present in 'other'.
        """
        self.states.update(other.states)
        self.widget_metadata.update(other.widget_metadata)

    def set_widget_from_proto(self, widget_state: WidgetStateProto) -> None:
        """Set a widget's serialized value, overwriting any existing value it has."""
        self[widget_state.id] = Serialized(widget_state)

    def set_from_value(self, k: str, v: Any) -> None:
        """Set a widget's deserialized value, overwriting any existing value it has."""
        self[k] = Value(v)

    def set_widget_metadata(self, widget_meta: WidgetMetadata[Any]) -> None:
        """Set a widget's metadata, overwriting any existing metadata it has."""
        self.widget_metadata[widget_meta.id] = widget_meta

    def remove_stale_widgets(self, active_widget_ids: set[str], fragment_ids_this_run: set[str] | None) -> None:
        """Remove widget state for stale widgets."""
        self.states = {k: v for k, v in self.states.items() if not _is_stale_widget(self.widget_metadata.get(k), active_widget_ids, fragment_ids_this_run)}

    def get_serialized(self, k: str) -> WidgetStateProto | None:
        """Get the serialized value of the widget with the given id.

        If the widget doesn't exist, return None. If the widget exists but
        is not in serialized form, it will be serialized first.
        """
        item = self.states.get(k)
        if item is None:
            return None
        if isinstance(item, Serialized):
            return item.value
        metadata = self.widget_metadata.get(k)
        if metadata is None:
            return None
        widget = WidgetStateProto()
        widget.id = k
        field = metadata.value_type
        serialized = metadata.serializer(item.value)
        if is_array_value_field_name(field):
            arr = getattr(widget, field)
            arr.data.extend(serialized)
        elif field == 'json_value':
            setattr(widget, field, json.dumps(serialized))
        elif field == 'file_uploader_state_value':
            widget.file_uploader_state_value.CopyFrom(serialized)
        elif field == 'string_trigger_value':
            widget.string_trigger_value.CopyFrom(serialized)
        elif field is not None and serialized is not None:
            setattr(widget, field, serialized)
        return widget

    def as_widget_states(self) -> list[WidgetStateProto]:
        """Return a list of serialized widget values for each widget with a value."""
        states = [self.get_serialized(widget_id) for widget_id in self.states.keys() if self.get_serialized(widget_id)]
        states = cast(List[WidgetStateProto], states)
        return states

    def call_callback(self, widget_id: str) -> None:
        """Call the given widget's callback and return the callback's
        return value. If the widget has no callback, return None.

        If the widget doesn't exist, raise an Exception.
        """
        metadata = self.widget_metadata.get(widget_id)
        assert metadata is not None
        callback = metadata.callback
        if callback is None:
            return
        args = metadata.callback_args or ()
        kwargs = metadata.callback_kwargs or {}
        callback(*args, **kwargs)