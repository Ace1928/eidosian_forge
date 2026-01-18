from __future__ import annotations
import glob
import os
import re
import pathlib
import shutil
import subprocess
import typing as T
import functools
from mesonbuild.interpreterbase.decorators import FeatureDeprecated
from .. import mesonlib, mlog
from ..environment import get_llvm_tool_names
from ..mesonlib import version_compare, version_compare_many, search_version, stringlistify, extract_as_list
from .base import DependencyException, DependencyMethods, detect_compiler, strip_system_includedirs, strip_system_libdirs, SystemDependency, ExternalDependency, DependencyTypeName
from .cmake import CMakeDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
from .misc import threads_factory
from .pkgconfig import PkgConfigDependency
@staticmethod
def __machine_info_to_platform_include_dir(m: 'MachineInfo') -> T.Optional[str]:
    """Translates the machine information to the platform-dependent include directory

        When inspecting a JDK release tarball or $JAVA_HOME, inside the `include/` directory is a
        platform-dependent directory that must be on the target's include path in addition to the
        parent `include/` directory.
        """
    if m.is_linux():
        return 'linux'
    elif m.is_windows():
        return 'win32'
    elif m.is_darwin():
        return 'darwin'
    elif m.is_sunos():
        return 'solaris'
    elif m.is_freebsd():
        return 'freebsd'
    elif m.is_netbsd():
        return 'netbsd'
    elif m.is_openbsd():
        return 'openbsd'
    elif m.is_dragonflybsd():
        return 'dragonfly'
    return None