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
def add_compiler_options(self, options: MutableKeyedOptionDictType, lang: str, for_machine: MachineChoice, env: Environment, subproject: str) -> None:
    for k, o in options.items():
        value = env.options.get(k)
        if value is not None:
            o.set_value(value)
            if not subproject:
                self.options[k] = o
        self.options.setdefault(k, o)
        if subproject:
            sk = k.evolve(subproject=subproject)
            value = env.options.get(sk) or value
            if value is not None:
                o.set_value(value)
                self.options[sk] = o
            self.options.setdefault(sk, o)