from __future__ import annotations
import os
import sys
from typing import TypeVar, Union
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import AbstractPointPen
class HasIdentifier(Protocol):
    """Any object that has a unique identifier in some context that can be
    used as a key in a public.objectLibs dictionary."""
    identifier: str | None