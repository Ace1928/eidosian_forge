from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAudio.nodeParamConnected')
@dataclass
class NodeParamConnected:
    """
    Notifies that an AudioNode is connected to an AudioParam.
    """
    context_id: GraphObjectId
    source_id: GraphObjectId
    destination_id: GraphObjectId
    source_output_index: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodeParamConnected:
        return cls(context_id=GraphObjectId.from_json(json['contextId']), source_id=GraphObjectId.from_json(json['sourceId']), destination_id=GraphObjectId.from_json(json['destinationId']), source_output_index=float(json['sourceOutputIndex']) if 'sourceOutputIndex' in json else None)