import io
import re
import functools
import inspect
import os
import sys
import numbers
import warnings
from pathlib import Path, PurePath
from typing import (
from ase.atoms import Atoms
from importlib import import_module
from ase.parallel import parallel_function, parallel_generator
def _warn_none(self, action):
    msg = 'Accessing the IOFormat.{action} property on a format without {action} support will change behaviour in the future and return a callable instead of None.  Use IOFormat.can_{action} to check whether {action} is supported.'
    warnings.warn(msg.format(action=action), FutureWarning)