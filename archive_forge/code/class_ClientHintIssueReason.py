from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
class ClientHintIssueReason(enum.Enum):
    META_TAG_ALLOW_LIST_INVALID_ORIGIN = 'MetaTagAllowListInvalidOrigin'
    META_TAG_MODIFIED_HTML = 'MetaTagModifiedHTML'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)