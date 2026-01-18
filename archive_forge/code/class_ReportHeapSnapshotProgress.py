from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@event_class('HeapProfiler.reportHeapSnapshotProgress')
@dataclass
class ReportHeapSnapshotProgress:
    done: int
    total: int
    finished: typing.Optional[bool]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportHeapSnapshotProgress:
        return cls(done=int(json['done']), total=int(json['total']), finished=bool(json['finished']) if 'finished' in json else None)