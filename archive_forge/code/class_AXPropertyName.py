from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
class AXPropertyName(enum.Enum):
    """
    Values of AXProperty name:
    - from 'busy' to 'roledescription': states which apply to every AX node
    - from 'live' to 'root': attributes which apply to nodes in live regions
    - from 'autocomplete' to 'valuetext': attributes which apply to widgets
    - from 'checked' to 'selected': states which apply to widgets
    - from 'activedescendant' to 'owns' - relationships between elements other than parent/child/sibling.
    """
    BUSY = 'busy'
    DISABLED = 'disabled'
    EDITABLE = 'editable'
    FOCUSABLE = 'focusable'
    FOCUSED = 'focused'
    HIDDEN = 'hidden'
    HIDDEN_ROOT = 'hiddenRoot'
    INVALID = 'invalid'
    KEYSHORTCUTS = 'keyshortcuts'
    SETTABLE = 'settable'
    ROLEDESCRIPTION = 'roledescription'
    LIVE = 'live'
    ATOMIC = 'atomic'
    RELEVANT = 'relevant'
    ROOT = 'root'
    AUTOCOMPLETE = 'autocomplete'
    HAS_POPUP = 'hasPopup'
    LEVEL = 'level'
    MULTISELECTABLE = 'multiselectable'
    ORIENTATION = 'orientation'
    MULTILINE = 'multiline'
    READONLY = 'readonly'
    REQUIRED = 'required'
    VALUEMIN = 'valuemin'
    VALUEMAX = 'valuemax'
    VALUETEXT = 'valuetext'
    CHECKED = 'checked'
    EXPANDED = 'expanded'
    MODAL = 'modal'
    PRESSED = 'pressed'
    SELECTED = 'selected'
    ACTIVEDESCENDANT = 'activedescendant'
    CONTROLS = 'controls'
    DESCRIBEDBY = 'describedby'
    DETAILS = 'details'
    ERRORMESSAGE = 'errormessage'
    FLOWTO = 'flowto'
    LABELLEDBY = 'labelledby'
    OWNS = 'owns'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)