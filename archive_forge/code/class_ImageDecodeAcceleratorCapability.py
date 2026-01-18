from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class ImageDecodeAcceleratorCapability:
    """
    Describes a supported image decoding profile with its associated minimum and
    maximum resolutions and subsampling.
    """
    image_type: ImageType
    max_dimensions: Size
    min_dimensions: Size
    subsamplings: typing.List[SubsamplingFormat]

    def to_json(self):
        json = dict()
        json['imageType'] = self.image_type.to_json()
        json['maxDimensions'] = self.max_dimensions.to_json()
        json['minDimensions'] = self.min_dimensions.to_json()
        json['subsamplings'] = [i.to_json() for i in self.subsamplings]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(image_type=ImageType.from_json(json['imageType']), max_dimensions=Size.from_json(json['maxDimensions']), min_dimensions=Size.from_json(json['minDimensions']), subsamplings=[SubsamplingFormat.from_json(i) for i in json['subsamplings']])