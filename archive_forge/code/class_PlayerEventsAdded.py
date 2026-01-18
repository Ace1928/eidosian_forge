from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Media.playerEventsAdded')
@dataclass
class PlayerEventsAdded:
    """
    Send events as a list, allowing them to be batched on the browser for less
    congestion. If batched, events must ALWAYS be in chronological order.
    """
    player_id: PlayerId
    events: typing.List[PlayerEvent]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerEventsAdded:
        return cls(player_id=PlayerId.from_json(json['playerId']), events=[PlayerEvent.from_json(i) for i in json['events']])