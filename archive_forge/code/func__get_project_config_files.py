import io
import itertools
import numbers
import os
import re
import sys
from contextlib import suppress
from glob import iglob
from pathlib import Path
from typing import List, Optional, Set
import distutils.cmd
import distutils.command
import distutils.core
import distutils.dist
import distutils.log
from distutils.debug import DEBUG
from distutils.errors import DistutilsOptionError, DistutilsSetupError
from distutils.fancy_getopt import translate_longopt
from distutils.util import strtobool
from .extern.more_itertools import partition, unique_everseen
from .extern.ordered_set import OrderedSet
from .extern.packaging.markers import InvalidMarker, Marker
from .extern.packaging.specifiers import InvalidSpecifier, SpecifierSet
from .extern.packaging.version import Version
from . import _entry_points
from . import _normalization
from . import _reqs
from . import command as _  # noqa  -- imported for side-effects
from ._importlib import metadata
from .config import setupcfg, pyprojecttoml
from .discovery import ConfigDiscovery
from .monkey import get_unpatched
from .warnings import InformationOnly, SetuptoolsDeprecationWarning
def _get_project_config_files(self, filenames):
    """Add default file and split between INI and TOML"""
    tomlfiles = []
    standard_project_metadata = Path(self.src_root or os.curdir, 'pyproject.toml')
    if filenames is not None:
        parts = partition(lambda f: Path(f).suffix == '.toml', filenames)
        filenames = list(parts[0])
        tomlfiles = list(parts[1])
    elif standard_project_metadata.exists():
        tomlfiles = [standard_project_metadata]
    return (filenames, tomlfiles)