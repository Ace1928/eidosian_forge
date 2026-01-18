from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class PrerenderMismatchedHeaders:
    """
    Information of headers to be displayed when the header mismatch occurred.
    """
    header_name: str
    initial_value: typing.Optional[str] = None
    activation_value: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['headerName'] = self.header_name
        if self.initial_value is not None:
            json['initialValue'] = self.initial_value
        if self.activation_value is not None:
            json['activationValue'] = self.activation_value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(header_name=str(json['headerName']), initial_value=str(json['initialValue']) if 'initialValue' in json else None, activation_value=str(json['activationValue']) if 'activationValue' in json else None)