from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class KeyModifiers(TypedDict):
    shift: bool
    ctrl: bool
    alt: bool