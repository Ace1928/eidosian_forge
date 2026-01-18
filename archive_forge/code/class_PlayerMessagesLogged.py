from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Media.playerMessagesLogged')
@dataclass
class PlayerMessagesLogged:
    """
    Send a list of any messages that need to be delivered.
    """
    player_id: PlayerId
    messages: typing.List[PlayerMessage]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerMessagesLogged:
        return cls(player_id=PlayerId.from_json(json['playerId']), messages=[PlayerMessage.from_json(i) for i in json['messages']])