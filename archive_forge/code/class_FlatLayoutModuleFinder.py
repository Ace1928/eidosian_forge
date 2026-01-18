import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
class FlatLayoutModuleFinder(ModuleFinder):
    DEFAULT_EXCLUDE = ('setup', 'conftest', 'test', 'tests', 'example', 'examples', 'build', 'toxfile', 'noxfile', 'pavement', 'dodo', 'tasks', 'fabfile', '[Ss][Cc]onstruct', 'conanfile', 'manage', 'benchmark', 'benchmarks', 'exercise', 'exercises', '[._]*')
    'Reserved top-level module names'