from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
@event_class('Profiler.preciseCoverageDeltaUpdate')
@dataclass
class PreciseCoverageDeltaUpdate:
    """
    **EXPERIMENTAL**

    Reports coverage delta since the last poll (either from an event like this, or from
    ``takePreciseCoverage`` for the current isolate. May only be sent if precise code
    coverage has been started. This event can be trigged by the embedder to, for example,
    trigger collection of coverage data immediatelly at a certain point in time.
    """
    timestamp: float
    occassion: str
    result: typing.List[ScriptCoverage]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PreciseCoverageDeltaUpdate:
        return cls(timestamp=float(json['timestamp']), occassion=str(json['occassion']), result=[ScriptCoverage.from_json(i) for i in json['result']])