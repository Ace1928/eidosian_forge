from __future__ import annotations
import dataclasses
import glob
import importlib
import itertools
import json
import os
import shutil
import collections
import typing as T
from . import builder
from . import version
from ..mesonlib import MesonException, Popen_safe, OptionKey
from .. import coredata
@dataclasses.dataclass
class BuildTarget:
    name: str
    crate_type: T.List[manifest.CRATE_TYPE] = dataclasses.field(default_factory=lambda: ['lib'])
    path: dataclasses.InitVar[T.Optional[str]] = None
    test: bool = True
    doctest: bool = False
    bench: bool = True
    doc: bool = False
    harness: bool = True
    edition: manifest.EDITION = '2015'
    required_features: T.List[str] = dataclasses.field(default_factory=list)
    plugin: bool = False