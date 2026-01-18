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
class UserStringOption(UserOption[str]):

    def __init__(self, description: str, value: T.Any, yielding: bool=DEFAULT_YIELDING, deprecated: T.Union[bool, str, T.Dict[str, str], T.List[str]]=False):
        super().__init__(description, None, yielding, deprecated)
        self.set_value(value)

    def validate_value(self, value: T.Any) -> str:
        if not isinstance(value, str):
            raise MesonException('Value "%s" for string option is not a string.' % str(value))
        return value