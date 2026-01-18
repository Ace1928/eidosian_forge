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
@event_class('Page.interstitialHidden')
@dataclass
class InterstitialHidden:
    """
    Fired when interstitial page was hidden
    """

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterstitialHidden:
        return cls()