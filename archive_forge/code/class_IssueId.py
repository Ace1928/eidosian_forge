from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
class IssueId(str):
    """
    A unique id for a DevTools inspector issue. Allows other entities (e.g.
    exceptions, CDP message, console messages, etc.) to reference an issue.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> IssueId:
        return cls(json)

    def __repr__(self):
        return 'IssueId({})'.format(super().__repr__())