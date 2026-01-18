from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Runtime.executionContextCreated')
@dataclass
class ExecutionContextCreated:
    """
    Issued when new execution context is created.
    """
    context: ExecutionContextDescription

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExecutionContextCreated:
        return cls(context=ExecutionContextDescription.from_json(json['context']))