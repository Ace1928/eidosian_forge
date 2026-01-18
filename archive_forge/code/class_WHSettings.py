from __future__ import annotations
import dataclasses
import enum
import typing
class WHSettings(str, enum.Enum):
    """Width and Height settings"""
    PACK = 'pack'
    GIVEN = 'given'
    RELATIVE = 'relative'
    WEIGHT = 'weight'
    CLIP = 'clip'
    FLOW = 'flow'