from __future__ import annotations
import functools
import re
import typing as T
from .. import mesonlib
from .. import mlog
from .base import DependencyException, DependencyMethods
from .base import BuiltinDependency, SystemDependency
from .cmake import CMakeDependency, CMakeDependencyFactory
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory, factory_methods
from .pkgconfig import PkgConfigDependency
class CursesSystemDependency(SystemDependency):
    """Curses dependency the hard way.

    This replaces hand rolled find_library() and has_header() calls. We
    provide this for portability reasons, there are a large number of curses
    implementations, and the differences between them can be very annoying.
    """

    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any]):
        super().__init__(name, env, kwargs)
        candidates = [('pdcurses', ['pdcurses/curses.h']), ('ncursesw', ['ncursesw/ncurses.h', 'ncurses.h']), ('ncurses', ['ncurses/ncurses.h', 'ncurses/curses.h', 'ncurses.h']), ('curses', ['curses.h'])]
        for lib, headers in candidates:
            l = self.clib_compiler.find_library(lib, env, [])
            if l:
                for header in headers:
                    h = self.clib_compiler.has_header(header, '', env)
                    if h[0]:
                        self.is_found = True
                        self.link_args = l
                        if lib.startswith('ncurses'):
                            v, _ = self.clib_compiler.get_define('NCURSES_VERSION', f'#include <{header}>', env, [], [self])
                            self.version = v.strip('"')
                        if lib.startswith('pdcurses'):
                            v_major, _ = self.clib_compiler.get_define('PDC_VER_MAJOR', f'#include <{header}>', env, [], [self])
                            v_minor, _ = self.clib_compiler.get_define('PDC_VER_MINOR', f'#include <{header}>', env, [], [self])
                            self.version = f'{v_major}.{v_minor}'
                        req = kwargs.get('version')
                        if req:
                            if self.version:
                                self.is_found = mesonlib.version_compare(self.version, req)
                            else:
                                mlog.warning('Cannot determine version of curses to compare against.')
                        if self.is_found:
                            mlog.debug('Curses library:', l)
                            mlog.debug('Curses header:', header)
                            break
            if self.is_found:
                break