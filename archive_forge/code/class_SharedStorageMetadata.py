from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class SharedStorageMetadata:
    """
    Details for an origin's shared storage.
    """
    creation_time: network.TimeSinceEpoch
    length: int
    remaining_budget: float

    def to_json(self):
        json = dict()
        json['creationTime'] = self.creation_time.to_json()
        json['length'] = self.length
        json['remainingBudget'] = self.remaining_budget
        return json

    @classmethod
    def from_json(cls, json):
        return cls(creation_time=network.TimeSinceEpoch.from_json(json['creationTime']), length=int(json['length']), remaining_budget=float(json['remainingBudget']))