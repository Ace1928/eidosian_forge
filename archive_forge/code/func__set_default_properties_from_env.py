from __future__ import annotations
import itertools
import os, platform, re, sys, shutil
import typing as T
import collections
from . import coredata
from . import mesonlib
from .mesonlib import (
from . import mlog
from .programs import ExternalProgram
from .envconfig import (
from . import compilers
from .compilers import (
from functools import lru_cache
from mesonbuild import envconfig
def _set_default_properties_from_env(self) -> None:
    """Properties which can also be set from the environment."""
    opts: T.List[T.Tuple[str, T.List[str], bool]] = [('boost_includedir', ['BOOST_INCLUDEDIR'], False), ('boost_librarydir', ['BOOST_LIBRARYDIR'], False), ('boost_root', ['BOOST_ROOT', 'BOOSTROOT'], True), ('java_home', ['JAVA_HOME'], False)]
    for (name, evars, split), for_machine in itertools.product(opts, MachineChoice):
        for evar in evars:
            p_env = _get_env_var(for_machine, self.is_cross_build(), evar)
            if p_env is not None:
                if split:
                    self.properties[for_machine].properties.setdefault(name, p_env.split(os.pathsep))
                else:
                    self.properties[for_machine].properties.setdefault(name, p_env)
                break