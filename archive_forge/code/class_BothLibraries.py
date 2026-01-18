from __future__ import annotations
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, InitVar
from functools import lru_cache
import abc
import hashlib
import itertools, pathlib
import os
import pickle
import re
import textwrap
import typing as T
from . import coredata
from . import dependencies
from . import mlog
from . import programs
from .mesonlib import (
from .compilers import (
from .interpreterbase import FeatureNew, FeatureDeprecated
class BothLibraries(SecondLevelHolder):

    def __init__(self, shared: SharedLibrary, static: StaticLibrary) -> None:
        self._preferred_library = 'shared'
        self.shared = shared
        self.static = static
        self.subproject = self.shared.subproject

    def __repr__(self) -> str:
        return f'<BothLibraries: static={repr(self.static)}; shared={repr(self.shared)}>'

    def get_default_object(self) -> BuildTarget:
        if self._preferred_library == 'shared':
            return self.shared
        elif self._preferred_library == 'static':
            return self.static
        raise MesonBugException(f'self._preferred_library == "{self._preferred_library}" is neither "shared" nor "static".')