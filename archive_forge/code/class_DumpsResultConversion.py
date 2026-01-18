import builtins
import codecs
import enum
import io
import json
import os
import types
import typing
from typing import (
import attr
class DumpsResultConversion(enum.Enum):
    LeaveAsIs = enum.auto()
    EncodeToBytes = enum.auto()
    DecodeToString = enum.auto()