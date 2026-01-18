from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@dataclass
class AXNode:
    """
    A node in the accessibility tree.
    """
    node_id: AXNodeId
    ignored: bool
    ignored_reasons: typing.Optional[typing.List[AXProperty]] = None
    role: typing.Optional[AXValue] = None
    name: typing.Optional[AXValue] = None
    description: typing.Optional[AXValue] = None
    value: typing.Optional[AXValue] = None
    properties: typing.Optional[typing.List[AXProperty]] = None
    child_ids: typing.Optional[typing.List[AXNodeId]] = None
    backend_dom_node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['nodeId'] = self.node_id.to_json()
        json['ignored'] = self.ignored
        if self.ignored_reasons is not None:
            json['ignoredReasons'] = [i.to_json() for i in self.ignored_reasons]
        if self.role is not None:
            json['role'] = self.role.to_json()
        if self.name is not None:
            json['name'] = self.name.to_json()
        if self.description is not None:
            json['description'] = self.description.to_json()
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.properties is not None:
            json['properties'] = [i.to_json() for i in self.properties]
        if self.child_ids is not None:
            json['childIds'] = [i.to_json() for i in self.child_ids]
        if self.backend_dom_node_id is not None:
            json['backendDOMNodeId'] = self.backend_dom_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(node_id=AXNodeId.from_json(json['nodeId']), ignored=bool(json['ignored']), ignored_reasons=[AXProperty.from_json(i) for i in json['ignoredReasons']] if 'ignoredReasons' in json else None, role=AXValue.from_json(json['role']) if 'role' in json else None, name=AXValue.from_json(json['name']) if 'name' in json else None, description=AXValue.from_json(json['description']) if 'description' in json else None, value=AXValue.from_json(json['value']) if 'value' in json else None, properties=[AXProperty.from_json(i) for i in json['properties']] if 'properties' in json else None, child_ids=[AXNodeId.from_json(i) for i in json['childIds']] if 'childIds' in json else None, backend_dom_node_id=dom.BackendNodeId.from_json(json['backendDOMNodeId']) if 'backendDOMNodeId' in json else None)