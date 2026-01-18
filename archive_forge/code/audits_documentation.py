from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page

    Enables issues domain, sends the issues collected so far to the client by means of the
    ``issueAdded`` event.
    