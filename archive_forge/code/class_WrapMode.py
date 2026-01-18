from __future__ import annotations
import dataclasses
import enum
import typing
class WrapMode(str, enum.Enum):
    """Text wrapping modes"""
    SPACE = 'space'
    ANY = 'any'
    CLIP = 'clip'
    ELLIPSIS = 'ellipsis'