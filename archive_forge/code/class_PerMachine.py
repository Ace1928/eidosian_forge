from __future__ import annotations
from pathlib import Path
import argparse
import enum
import sys
import stat
import time
import abc
import platform, subprocess, operator, os, shlex, shutil, re
import collections
from functools import lru_cache, wraps, total_ordering
from itertools import tee
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing as T
import textwrap
import pickle
import errno
import json
from mesonbuild import mlog
from .core import MesonException, HoldableObject
from glob import glob
class PerMachine(T.Generic[_T]):

    def __init__(self, build: _T, host: _T) -> None:
        self.build = build
        self.host = host

    def __getitem__(self, machine: MachineChoice) -> _T:
        return {MachineChoice.BUILD: self.build, MachineChoice.HOST: self.host}[machine]

    def __setitem__(self, machine: MachineChoice, val: _T) -> None:
        setattr(self, machine.get_lower_case_name(), val)

    def miss_defaulting(self) -> 'PerMachineDefaultable[T.Optional[_T]]':
        """Unset definition duplicated from their previous to None

        This is the inverse of ''default_missing''. By removing defaulted
        machines, we can elaborate the original and then redefault them and thus
        avoid repeating the elaboration explicitly.
        """
        unfreeze: PerMachineDefaultable[T.Optional[_T]] = PerMachineDefaultable()
        unfreeze.build = self.build
        unfreeze.host = self.host
        if unfreeze.host == unfreeze.build:
            unfreeze.host = None
        return unfreeze

    def assign(self, build: _T, host: _T) -> None:
        self.build = build
        self.host = host

    def __repr__(self) -> str:
        return f'PerMachine({self.build!r}, {self.host!r})'