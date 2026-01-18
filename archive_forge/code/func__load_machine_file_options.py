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
def _load_machine_file_options(self, config: 'ConfigParser', properties: Properties, machine: MachineChoice) -> None:
    """Read the contents of a Machine file and put it in the options store."""
    paths = config.get('paths')
    if paths:
        mlog.deprecation('The [paths] section is deprecated, use the [built-in options] section instead.')
        for k, v in paths.items():
            self.options[OptionKey.from_string(k).evolve(machine=machine)] = v
    deprecated_properties: T.Set[str] = set()
    for lang in compilers.all_languages:
        deprecated_properties.add(lang + '_args')
        deprecated_properties.add(lang + '_link_args')
    for k, v in properties.properties.copy().items():
        if k in deprecated_properties:
            mlog.deprecation(f'{k} in the [properties] section of the machine file is deprecated, use the [built-in options] section.')
            self.options[OptionKey.from_string(k).evolve(machine=machine)] = v
            del properties.properties[k]
    for section, values in config.items():
        if ':' in section:
            subproject, section = section.split(':')
        else:
            subproject = ''
        if section == 'built-in options':
            for k, v in values.items():
                key = OptionKey.from_string(k)
                if machine is MachineChoice.HOST and key.machine is not machine:
                    mlog.deprecation('Setting build machine options in cross files, please use a native file instead, this will be removed in meson 0.60', once=True)
                if key.subproject:
                    raise MesonException('Do not set subproject options in [built-in options] section, use [subproject:built-in options] instead.')
                self.options[key.evolve(subproject=subproject, machine=machine)] = v
        elif section == 'project options' and machine is MachineChoice.HOST:
            for k, v in values.items():
                key = OptionKey.from_string(k)
                if key.subproject:
                    raise MesonException('Do not set subproject options in [built-in options] section, use [subproject:built-in options] instead.')
                self.options[key.evolve(subproject=subproject)] = v