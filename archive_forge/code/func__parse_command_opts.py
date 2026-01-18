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
def _parse_command_opts(self, parser, args):
    self.global_options = self.__class__.global_options
    self.negative_opt = self.__class__.negative_opt
    command = args[0]
    aliases = self.get_option_dict('aliases')
    while command in aliases:
        src, alias = aliases[command]
        del aliases[command]
        import shlex
        args[:1] = shlex.split(alias, True)
        command = args[0]
    nargs = _Distribution._parse_command_opts(self, parser, args)
    cmd_class = self.get_command_class(command)
    if getattr(cmd_class, 'command_consumes_arguments', None):
        self.get_option_dict(command)['args'] = ('command line', nargs)
        if nargs is not None:
            return []
    return nargs