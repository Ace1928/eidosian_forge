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
class AdFrameStatus:
    """
    Indicates whether a frame has been identified as an ad and why.
    """
    ad_frame_type: AdFrameType
    explanations: typing.Optional[typing.List[AdFrameExplanation]] = None

    def to_json(self):
        json = dict()
        json['adFrameType'] = self.ad_frame_type.to_json()
        if self.explanations is not None:
            json['explanations'] = [i.to_json() for i in self.explanations]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(ad_frame_type=AdFrameType.from_json(json['adFrameType']), explanations=[AdFrameExplanation.from_json(i) for i in json['explanations']] if 'explanations' in json else None)