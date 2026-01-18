from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class StackTraceId:
    """
    If ``debuggerId`` is set stack trace comes from another debugger and can be resolved there. This
    allows to track cross-debugger calls. See ``Runtime.StackTrace`` and ``Debugger.paused`` for usages.
    """
    id_: str
    debugger_id: typing.Optional[UniqueDebuggerId] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_
        if self.debugger_id is not None:
            json['debuggerId'] = self.debugger_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(id_=str(json['id']), debugger_id=UniqueDebuggerId.from_json(json['debuggerId']) if 'debuggerId' in json else None)