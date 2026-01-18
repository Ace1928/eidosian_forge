from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
class PseudoType(enum.Enum):
    """
    Pseudo element type.
    """
    FIRST_LINE = 'first-line'
    FIRST_LETTER = 'first-letter'
    BEFORE = 'before'
    AFTER = 'after'
    MARKER = 'marker'
    BACKDROP = 'backdrop'
    SELECTION = 'selection'
    TARGET_TEXT = 'target-text'
    SPELLING_ERROR = 'spelling-error'
    GRAMMAR_ERROR = 'grammar-error'
    HIGHLIGHT = 'highlight'
    FIRST_LINE_INHERITED = 'first-line-inherited'
    SCROLLBAR = 'scrollbar'
    SCROLLBAR_THUMB = 'scrollbar-thumb'
    SCROLLBAR_BUTTON = 'scrollbar-button'
    SCROLLBAR_TRACK = 'scrollbar-track'
    SCROLLBAR_TRACK_PIECE = 'scrollbar-track-piece'
    SCROLLBAR_CORNER = 'scrollbar-corner'
    RESIZER = 'resizer'
    INPUT_LIST_BUTTON = 'input-list-button'
    VIEW_TRANSITION = 'view-transition'
    VIEW_TRANSITION_GROUP = 'view-transition-group'
    VIEW_TRANSITION_IMAGE_PAIR = 'view-transition-image-pair'
    VIEW_TRANSITION_OLD = 'view-transition-old'
    VIEW_TRANSITION_NEW = 'view-transition-new'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)