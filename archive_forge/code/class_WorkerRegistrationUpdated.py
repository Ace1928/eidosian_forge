from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
@event_class('ServiceWorker.workerRegistrationUpdated')
@dataclass
class WorkerRegistrationUpdated:
    registrations: typing.List[ServiceWorkerRegistration]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerRegistrationUpdated:
        return cls(registrations=[ServiceWorkerRegistration.from_json(i) for i in json['registrations']])