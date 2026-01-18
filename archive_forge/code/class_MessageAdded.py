from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Console.messageAdded')
@dataclass
class MessageAdded:
    """
    Issued when new console message is added.
    """
    message: ConsoleMessage

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> MessageAdded:
        return cls(message=ConsoleMessage.from_json(json['message']))