from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Media.playersCreated')
@dataclass
class PlayersCreated:
    """
    Called whenever a player is created, or when a new agent joins and recieves
    a list of active players. If an agent is restored, it will recieve the full
    list of player ids and all events again.
    """
    players: typing.List[PlayerId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayersCreated:
        return cls(players=[PlayerId.from_json(i) for i in json['players']])