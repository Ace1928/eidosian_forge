from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
@event_class('Target.receivedMessageFromTarget')
@dataclass
class ReceivedMessageFromTarget:
    """
    Notifies about a new protocol message received from the session (as reported in
    ``attachedToTarget`` event).
    """
    session_id: SessionID
    message: str
    target_id: typing.Optional[TargetID]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReceivedMessageFromTarget:
        return cls(session_id=SessionID.from_json(json['sessionId']), message=str(json['message']), target_id=TargetID.from_json(json['targetId']) if 'targetId' in json else None)