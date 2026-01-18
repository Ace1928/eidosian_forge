from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page
@dataclass
class InlineTextBox:
    """
    Details of post layout rendered text positions. The exact layout should not be regarded as
    stable and may change between versions.
    """
    bounding_box: dom.Rect
    start_character_index: int
    num_characters: int

    def to_json(self):
        json = dict()
        json['boundingBox'] = self.bounding_box.to_json()
        json['startCharacterIndex'] = self.start_character_index
        json['numCharacters'] = self.num_characters
        return json

    @classmethod
    def from_json(cls, json):
        return cls(bounding_box=dom.Rect.from_json(json['boundingBox']), start_character_index=int(json['startCharacterIndex']), num_characters=int(json['numCharacters']))