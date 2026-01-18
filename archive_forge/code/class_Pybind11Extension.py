import contextlib
import os
import platform
import shlex
import shutil
import sys
import sysconfig
import tempfile
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import (
import distutils.ccompiler
import distutils.errors
class Pybind11Extension(_Extension):
    """
    Build a C++11+ Extension module with pybind11. This automatically adds the
    recommended flags when you init the extension and assumes C++ sources - you
    can further modify the options yourself.

    The customizations are:

    * ``/EHsc`` and ``/bigobj`` on Windows
    * ``stdlib=libc++`` on macOS
    * ``visibility=hidden`` and ``-g0`` on Unix

    Finally, you can set ``cxx_std`` via constructor or afterwards to enable
    flags for C++ std, and a few extra helper flags related to the C++ standard
    level. It is _highly_ recommended you either set this, or use the provided
    ``build_ext``, which will search for the highest supported extension for
    you if the ``cxx_std`` property is not set. Do not set the ``cxx_std``
    property more than once, as flags are added when you set it. Set the
    property to None to disable the addition of C++ standard flags.

    If you want to add pybind11 headers manually, for example for an exact
    git checkout, then set ``include_pybind11=False``.
    """

    def _add_cflags(self, flags: List[str]) -> None:
        self.extra_compile_args[:0] = flags

    def _add_ldflags(self, flags: List[str]) -> None:
        self.extra_link_args[:0] = flags

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._cxx_level = 0
        cxx_std = kwargs.pop('cxx_std', 0)
        if 'language' not in kwargs:
            kwargs['language'] = 'c++'
        include_pybind11 = kwargs.pop('include_pybind11', True)
        super().__init__(*args, **kwargs)
        if include_pybind11:
            try:
                import pybind11
                pyinc = pybind11.get_include()
                if pyinc not in self.include_dirs:
                    self.include_dirs.append(pyinc)
            except ModuleNotFoundError:
                pass
        self.cxx_std = cxx_std
        cflags = []
        if WIN:
            cflags += ['/EHsc', '/bigobj']
        else:
            cflags += ['-fvisibility=hidden']
            env_cflags = os.environ.get('CFLAGS', '')
            env_cppflags = os.environ.get('CPPFLAGS', '')
            c_cpp_flags = shlex.split(env_cflags) + shlex.split(env_cppflags)
            if not any((opt.startswith('-g') for opt in c_cpp_flags)):
                cflags += ['-g0']
        self._add_cflags(cflags)

    @property
    def cxx_std(self) -> int:
        """
        The CXX standard level. If set, will add the required flags. If left at
        0, it will trigger an automatic search when pybind11's build_ext is
        used. If None, will have no effect.  Besides just the flags, this may
        add a macos-min 10.9 or 10.14 flag if MACOSX_DEPLOYMENT_TARGET is
        unset.
        """
        return self._cxx_level

    @cxx_std.setter
    def cxx_std(self, level: int) -> None:
        if self._cxx_level:
            warnings.warn('You cannot safely change the cxx_level after setting it!', stacklevel=2)
        if WIN and level == 11:
            level = 14
        self._cxx_level = level
        if not level:
            return
        cflags = [STD_TMPL.format(level)]
        ldflags = []
        if MACOS and 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
            current_macos = tuple((int(x) for x in platform.mac_ver()[0].split('.')[:2]))
            desired_macos = (10, 9) if level < 17 else (10, 14)
            macos_string = '.'.join((str(x) for x in min(current_macos, desired_macos)))
            macosx_min = f'-mmacosx-version-min={macos_string}'
            cflags += [macosx_min]
            ldflags += [macosx_min]
        self._add_cflags(cflags)
        self._add_ldflags(ldflags)