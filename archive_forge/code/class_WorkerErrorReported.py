from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
@event_class('ServiceWorker.workerErrorReported')
@dataclass
class WorkerErrorReported:
    error_message: ServiceWorkerErrorMessage

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerErrorReported:
        return cls(error_message=ServiceWorkerErrorMessage.from_json(json['errorMessage']))