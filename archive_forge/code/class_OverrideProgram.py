from __future__ import annotations
import functools
import os
import shutil
import stat
import sys
import re
import typing as T
from pathlib import Path
from . import mesonlib
from . import mlog
from .mesonlib import MachineChoice, OrderedSet
class OverrideProgram(ExternalProgram):
    """A script overriding a program."""