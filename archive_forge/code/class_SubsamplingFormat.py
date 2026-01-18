from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class SubsamplingFormat(enum.Enum):
    """
    YUV subsampling type of the pixels of a given image.
    """
    YUV420 = 'yuv420'
    YUV422 = 'yuv422'
    YUV444 = 'yuv444'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)