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
class CmdLineFileParser(configparser.ConfigParser):

    def __init__(self) -> None:
        super().__init__(delimiters=['='], interpolation=None)

    def read(self, filenames: T.Union['StrOrBytesPath', T.Iterable['StrOrBytesPath']], encoding: T.Optional[str]='utf-8') -> T.List[str]:
        return super().read(filenames, encoding)

    def optionxform(self, optionstr: str) -> str:
        return optionstr