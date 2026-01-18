from __future__ import annotations
import dataclasses
import enum
import typing
class Sizing(str, enum.Enum):
    """Widget sizing methods."""
    FLOW = 'flow'
    BOX = 'box'
    FIXED = 'fixed'