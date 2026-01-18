from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
class TargetIntrospectionData(TypedDict):
    language: str
    compiler: T.List[str]
    parameters: T.List[str]
    sources: T.List[str]
    generated_sources: T.List[str]