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
def _bundle_static_library(self, t: T.Union[BuildTargetTypes], promoted: bool=False) -> None:
    if self.uses_rust():
        self.link_whole_targets.append(t)
    elif isinstance(t, (CustomTarget, CustomTargetIndex)) or t.uses_rust():
        m = f'Cannot link_whole a custom or Rust target {t.name!r} into a static library {self.name!r}. Instead, pass individual object files with the "objects:" keyword argument if possible.'
        if promoted:
            m += f' Meson had to promote link to link_whole because {self.name!r} is installed but not {t.name!r}, and thus has to include objects from {t.name!r} to be usable.'
        raise InvalidArguments(m)
    else:
        self.objects.append(t.extract_all_objects())