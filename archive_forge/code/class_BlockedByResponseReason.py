from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
class BlockedByResponseReason(enum.Enum):
    """
    Enum indicating the reason a response has been blocked. These reasons are
    refinements of the net error BLOCKED_BY_RESPONSE.
    """
    COEP_FRAME_RESOURCE_NEEDS_COEP_HEADER = 'CoepFrameResourceNeedsCoepHeader'
    COOP_SANDBOXED_I_FRAME_CANNOT_NAVIGATE_TO_COOP_PAGE = 'CoopSandboxedIFrameCannotNavigateToCoopPage'
    CORP_NOT_SAME_ORIGIN = 'CorpNotSameOrigin'
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_COEP = 'CorpNotSameOriginAfterDefaultedToSameOriginByCoep'
    CORP_NOT_SAME_SITE = 'CorpNotSameSite'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)