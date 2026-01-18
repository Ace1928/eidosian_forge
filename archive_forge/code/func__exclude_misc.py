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
def _exclude_misc(self, name, value):
    """Handle 'exclude()' for list/tuple attrs without a special handler"""
    if not isinstance(value, sequence):
        raise DistutilsSetupError('%s: setting must be a list or tuple (%r)' % (name, value))
    try:
        old = getattr(self, name)
    except AttributeError as e:
        raise DistutilsSetupError('%s: No such distribution setting' % name) from e
    if old is not None and (not isinstance(old, sequence)):
        raise DistutilsSetupError(name + ': this setting cannot be changed via include/exclude')
    elif old:
        setattr(self, name, [item for item in old if item not in value])