from __future__ import annotations
import copy
import os
import collections
import itertools
import typing as T
from enum import Enum
from .. import mlog, mesonlib
from ..compilers import clib_langs
from ..mesonlib import LibType, MachineChoice, MesonException, HoldableObject, OptionKey
from ..mesonlib import version_compare_many
class DependencyMethods(Enum):
    AUTO = 'auto'
    PKGCONFIG = 'pkg-config'
    CMAKE = 'cmake'
    BUILTIN = 'builtin'
    SYSTEM = 'system'
    EXTRAFRAMEWORK = 'extraframework'
    SYSCONFIG = 'sysconfig'
    CONFIG_TOOL = 'config-tool'
    SDLCONFIG = 'sdlconfig'
    CUPSCONFIG = 'cups-config'
    PCAPCONFIG = 'pcap-config'
    LIBWMFCONFIG = 'libwmf-config'
    QMAKE = 'qmake'
    DUB = 'dub'