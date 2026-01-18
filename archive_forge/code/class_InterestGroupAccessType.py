from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
class InterestGroupAccessType(enum.Enum):
    """
    Enum of interest group access types.
    """
    JOIN = 'join'
    LEAVE = 'leave'
    UPDATE = 'update'
    LOADED = 'loaded'
    BID = 'bid'
    WIN = 'win'
    ADDITIONAL_BID = 'additionalBid'
    ADDITIONAL_BID_WIN = 'additionalBidWin'
    TOP_LEVEL_BID = 'topLevelBid'
    TOP_LEVEL_ADDITIONAL_BID = 'topLevelAdditionalBid'
    CLEAR = 'clear'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)