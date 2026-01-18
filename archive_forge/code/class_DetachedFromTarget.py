from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
@event_class('Target.detachedFromTarget')
@dataclass
class DetachedFromTarget:
    """
    **EXPERIMENTAL**

    Issued when detached from target for any reason (including ``detachFromTarget`` command). Can be
    issued multiple times per target if multiple sessions have been attached to it.
    """
    session_id: SessionID
    target_id: typing.Optional[TargetID]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DetachedFromTarget:
        return cls(session_id=SessionID.from_json(json['sessionId']), target_id=TargetID.from_json(json['targetId']) if 'targetId' in json else None)