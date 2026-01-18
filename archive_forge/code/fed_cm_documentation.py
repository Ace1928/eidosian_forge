from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

    Triggered when a dialog is closed, either by user action, JS abort,
    or a command below.
    