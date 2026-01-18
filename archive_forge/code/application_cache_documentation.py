from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page

    Returns manifest URL for document in the given frame.

    :param frame_id: Identifier of the frame containing document whose manifest is retrieved.
    :returns: Manifest URL for document in the given frame.
    