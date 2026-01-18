from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@dataclass
class LayoutViewport:
    """
    Layout viewport position and dimensions.
    """
    page_x: int
    page_y: int
    client_width: int
    client_height: int

    def to_json(self):
        json = dict()
        json['pageX'] = self.page_x
        json['pageY'] = self.page_y
        json['clientWidth'] = self.client_width
        json['clientHeight'] = self.client_height
        return json

    @classmethod
    def from_json(cls, json):
        return cls(page_x=int(json['pageX']), page_y=int(json['pageY']), client_width=int(json['clientWidth']), client_height=int(json['clientHeight']))