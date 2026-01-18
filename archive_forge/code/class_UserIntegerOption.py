from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
class UserIntegerOption(UserOption[int]):

    def __init__(self, description: str, value: T.Any, yielding: bool=DEFAULT_YIELDING, deprecated: T.Union[bool, str, T.Dict[str, str], T.List[str]]=False):
        min_value, max_value, default_value = value
        self.min_value = min_value
        self.max_value = max_value
        c: T.List[str] = []
        if min_value is not None:
            c.append('>=' + str(min_value))
        if max_value is not None:
            c.append('<=' + str(max_value))
        choices = ', '.join(c)
        super().__init__(description, choices, yielding, deprecated)
        self.set_value(default_value)

    def validate_value(self, value: T.Any) -> int:
        if isinstance(value, str):
            value = self.toint(value)
        if not isinstance(value, int):
            raise MesonException('New value for integer option is not an integer.')
        if self.min_value is not None and value < self.min_value:
            raise MesonException('New value %d is less than minimum value %d.' % (value, self.min_value))
        if self.max_value is not None and value > self.max_value:
            raise MesonException('New value %d is more than maximum value %d.' % (value, self.max_value))
        return value

    def toint(self, valuestring: str) -> int:
        try:
            return int(valuestring)
        except ValueError:
            raise MesonException('Value string "%s" is not convertible to an integer.' % valuestring)