from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@dataclass
class AdScriptId:
    """
    Identifies the bottom-most script which caused the frame to be labelled
    as an ad.
    """
    script_id: runtime.ScriptId
    debugger_id: runtime.UniqueDebuggerId

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['debuggerId'] = self.debugger_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(script_id=runtime.ScriptId.from_json(json['scriptId']), debugger_id=runtime.UniqueDebuggerId.from_json(json['debuggerId']))