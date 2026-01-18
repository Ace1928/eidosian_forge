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
def _extract_pic_pie(self, kwargs: T.Dict[str, T.Any], arg: str, option: str) -> bool:
    all_flags = self.extra_args['c'] + self.extra_args['cpp']
    if '-f' + arg.lower() in all_flags or '-f' + arg.upper() in all_flags:
        mlog.warning(f"Use the '{arg}' kwarg instead of passing '-f{arg}' manually to {self.name!r}")
        return True
    k = OptionKey(option)
    if kwargs.get(arg) is not None:
        val = T.cast('bool', kwargs[arg])
    elif k in self.environment.coredata.options:
        val = self.environment.coredata.options[k].value
    else:
        val = False
    if not isinstance(val, bool):
        raise InvalidArguments(f'Argument {arg} to {self.name!r} must be boolean')
    return val