from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('FedCm.dialogClosed')
@dataclass
class DialogClosed:
    """
    Triggered when a dialog is closed, either by user action, JS abort,
    or a command below.
    """
    dialog_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DialogClosed:
        return cls(dialog_id=str(json['dialogId']))