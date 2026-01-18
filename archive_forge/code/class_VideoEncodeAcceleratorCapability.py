from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class VideoEncodeAcceleratorCapability:
    """
    Describes a supported video encoding profile with its associated maximum
    resolution and maximum framerate.
    """
    profile: str
    max_resolution: Size
    max_framerate_numerator: int
    max_framerate_denominator: int

    def to_json(self):
        json = dict()
        json['profile'] = self.profile
        json['maxResolution'] = self.max_resolution.to_json()
        json['maxFramerateNumerator'] = self.max_framerate_numerator
        json['maxFramerateDenominator'] = self.max_framerate_denominator
        return json

    @classmethod
    def from_json(cls, json):
        return cls(profile=str(json['profile']), max_resolution=Size.from_json(json['maxResolution']), max_framerate_numerator=int(json['maxFramerateNumerator']), max_framerate_denominator=int(json['maxFramerateDenominator']))