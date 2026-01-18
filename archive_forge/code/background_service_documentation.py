from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import service_worker

    Called with all existing backgroundServiceEvents when enabled, and all new
    events afterwards if enabled and recording.
    