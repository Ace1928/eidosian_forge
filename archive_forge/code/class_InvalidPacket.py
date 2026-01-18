from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
class InvalidPacket(Exception):
    pass