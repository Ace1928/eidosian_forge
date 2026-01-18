from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAudio.nodesDisconnected')
@dataclass
class NodesDisconnected:
    """
    Notifies that AudioNodes are disconnected. The destination can be null, and it means all the outgoing connections from the source are disconnected.
    """
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]
    destination_input_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodesDisconnected:
        return cls(context_id=GraphObjectId.from_json(json['contextId']), source_id=GraphObjectId.from_json(json['sourceId']), destination_id=GraphObjectId.from_json(json['destinationId']), source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None, destination_input_index=float(json['destinationInputIndex']) if 'destinationInputIndex' in json else None)