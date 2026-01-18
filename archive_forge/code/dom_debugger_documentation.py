from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime

    Sets breakpoint on XMLHttpRequest.

    :param url: Resource URL substring. All XHRs having this substring in the URL will get stopped upon.
    