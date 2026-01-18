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
def _set_others_from_buildtype(self, value: str) -> bool:
    dirty = False
    if value == 'plain':
        opt = 'plain'
        debug = False
    elif value == 'debug':
        opt = '0'
        debug = True
    elif value == 'debugoptimized':
        opt = '2'
        debug = True
    elif value == 'release':
        opt = '3'
        debug = False
    elif value == 'minsize':
        opt = 's'
        debug = True
    else:
        assert value == 'custom'
        return False
    dirty |= self.options[OptionKey('optimization')].set_value(opt)
    dirty |= self.options[OptionKey('debug')].set_value(debug)
    return dirty