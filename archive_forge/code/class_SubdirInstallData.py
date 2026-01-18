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
class SubdirInstallData(InstallDataBase):

    def __init__(self, path: str, install_path: str, install_path_name: str, install_mode: 'FileMode', exclude: T.Tuple[T.Set[str], T.Set[str]], subproject: str, tag: T.Optional[str]=None, data_type: T.Optional[str]=None, follow_symlinks: T.Optional[bool]=None):
        super().__init__(path, install_path, install_path_name, install_mode, subproject, tag, data_type, follow_symlinks)
        self.exclude = exclude