from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page
@dataclass
class NodeTreeSnapshot:
    """
    Table containing nodes.
    """
    parent_index: typing.Optional[typing.List[int]] = None
    node_type: typing.Optional[typing.List[int]] = None
    node_name: typing.Optional[typing.List[StringIndex]] = None
    node_value: typing.Optional[typing.List[StringIndex]] = None
    backend_node_id: typing.Optional[typing.List[dom.BackendNodeId]] = None
    attributes: typing.Optional[typing.List[ArrayOfStrings]] = None
    text_value: typing.Optional[RareStringData] = None
    input_value: typing.Optional[RareStringData] = None
    input_checked: typing.Optional[RareBooleanData] = None
    option_selected: typing.Optional[RareBooleanData] = None
    content_document_index: typing.Optional[RareIntegerData] = None
    pseudo_type: typing.Optional[RareStringData] = None
    is_clickable: typing.Optional[RareBooleanData] = None
    current_source_url: typing.Optional[RareStringData] = None
    origin_url: typing.Optional[RareStringData] = None

    def to_json(self):
        json = dict()
        if self.parent_index is not None:
            json['parentIndex'] = [i for i in self.parent_index]
        if self.node_type is not None:
            json['nodeType'] = [i for i in self.node_type]
        if self.node_name is not None:
            json['nodeName'] = [i.to_json() for i in self.node_name]
        if self.node_value is not None:
            json['nodeValue'] = [i.to_json() for i in self.node_value]
        if self.backend_node_id is not None:
            json['backendNodeId'] = [i.to_json() for i in self.backend_node_id]
        if self.attributes is not None:
            json['attributes'] = [i.to_json() for i in self.attributes]
        if self.text_value is not None:
            json['textValue'] = self.text_value.to_json()
        if self.input_value is not None:
            json['inputValue'] = self.input_value.to_json()
        if self.input_checked is not None:
            json['inputChecked'] = self.input_checked.to_json()
        if self.option_selected is not None:
            json['optionSelected'] = self.option_selected.to_json()
        if self.content_document_index is not None:
            json['contentDocumentIndex'] = self.content_document_index.to_json()
        if self.pseudo_type is not None:
            json['pseudoType'] = self.pseudo_type.to_json()
        if self.is_clickable is not None:
            json['isClickable'] = self.is_clickable.to_json()
        if self.current_source_url is not None:
            json['currentSourceURL'] = self.current_source_url.to_json()
        if self.origin_url is not None:
            json['originURL'] = self.origin_url.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(parent_index=[int(i) for i in json['parentIndex']] if 'parentIndex' in json else None, node_type=[int(i) for i in json['nodeType']] if 'nodeType' in json else None, node_name=[StringIndex.from_json(i) for i in json['nodeName']] if 'nodeName' in json else None, node_value=[StringIndex.from_json(i) for i in json['nodeValue']] if 'nodeValue' in json else None, backend_node_id=[dom.BackendNodeId.from_json(i) for i in json['backendNodeId']] if 'backendNodeId' in json else None, attributes=[ArrayOfStrings.from_json(i) for i in json['attributes']] if 'attributes' in json else None, text_value=RareStringData.from_json(json['textValue']) if 'textValue' in json else None, input_value=RareStringData.from_json(json['inputValue']) if 'inputValue' in json else None, input_checked=RareBooleanData.from_json(json['inputChecked']) if 'inputChecked' in json else None, option_selected=RareBooleanData.from_json(json['optionSelected']) if 'optionSelected' in json else None, content_document_index=RareIntegerData.from_json(json['contentDocumentIndex']) if 'contentDocumentIndex' in json else None, pseudo_type=RareStringData.from_json(json['pseudoType']) if 'pseudoType' in json else None, is_clickable=RareBooleanData.from_json(json['isClickable']) if 'isClickable' in json else None, current_source_url=RareStringData.from_json(json['currentSourceURL']) if 'currentSourceURL' in json else None, origin_url=RareStringData.from_json(json['originURL']) if 'originURL' in json else None)