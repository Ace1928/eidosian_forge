from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
@event_class('ServiceWorker.workerVersionUpdated')
@dataclass
class WorkerVersionUpdated:
    versions: typing.List[ServiceWorkerVersion]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerVersionUpdated:
        return cls(versions=[ServiceWorkerVersion.from_json(i) for i in json['versions']])