from __future__ import annotations
from pathlib import Path
from .base import ExternalDependency, DependencyException, sort_libpaths, DependencyTypeName
from ..mesonlib import EnvironmentVariables, OptionKey, OrderedSet, PerMachine, Popen_safe, Popen_safe_logged, MachineChoice, join_args
from ..programs import find_external_program, ExternalProgram
from .. import mlog
from pathlib import PurePath
from functools import lru_cache
import re
import os
import shlex
import typing as T
def _search_libs(self, libs_in: ImmutableListProtocol[str], raw_libs_in: ImmutableListProtocol[str]) -> T.Tuple[T.List[str], T.List[str]]:
    """
        @libs_in: PKG_CONFIG_ALLOW_SYSTEM_LIBS=1 pkg-config --libs
        @raw_libs_in: pkg-config --libs

        We always look for the file ourselves instead of depending on the
        compiler to find it with -lfoo or foo.lib (if possible) because:
        1. We want to be able to select static or shared
        2. We need the full path of the library to calculate RPATH values
        3. De-dup of libraries is easier when we have absolute paths

        Libraries that are provided by the toolchain or are not found by
        find_library() will be added with -L -l pairs.
        """
    prefix_libpaths: OrderedSet[str] = OrderedSet()
    raw_link_args = self._convert_mingw_paths(raw_libs_in)
    for arg in raw_link_args:
        if arg.startswith('-L') and (not arg.startswith(('-L-l', '-L-L'))):
            path = arg[2:]
            if not os.path.isabs(path):
                path = os.path.join(self.env.get_build_dir(), path)
            prefix_libpaths.add(path)
    pkg_config_path: T.List[str] = self.env.coredata.options[OptionKey('pkg_config_path', machine=self.for_machine)].value
    pkg_config_path = self._convert_mingw_paths(pkg_config_path)
    prefix_libpaths = OrderedSet(sort_libpaths(list(prefix_libpaths), pkg_config_path))
    system_libpaths: OrderedSet[str] = OrderedSet()
    full_args = self._convert_mingw_paths(libs_in)
    for arg in full_args:
        if arg.startswith(('-L-l', '-L-L')):
            continue
        if arg.startswith('-L') and arg[2:] not in prefix_libpaths:
            system_libpaths.add(arg[2:])
    libpaths = list(prefix_libpaths) + list(system_libpaths)
    libs_found: OrderedSet[str] = OrderedSet()
    libs_notfound = []
    link_args = []
    for lib in full_args:
        if lib.startswith(('-L-l', '-L-L')):
            pass
        elif lib.startswith('-L'):
            continue
        elif lib.startswith('-l:'):
            if lib in libs_found:
                continue
            libfilename = lib[3:]
            foundname = None
            for libdir in libpaths:
                target = os.path.join(libdir, libfilename)
                if os.path.exists(target):
                    foundname = target
                    break
            if foundname is None:
                if lib in libs_notfound:
                    continue
                else:
                    mlog.warning('Library {!r} not found for dependency {!r}, may not be successfully linked'.format(libfilename, self.name))
                libs_notfound.append(lib)
            else:
                lib = foundname
        elif lib.startswith('-l'):
            if lib in libs_found:
                continue
            if self.clib_compiler:
                args = self.clib_compiler.find_library(lib[2:], self.env, libpaths, self.libtype, lib_prefix_warning=False)
            else:
                args = None
            if args is not None:
                libs_found.add(lib)
                if args:
                    if not args[0].startswith('-l'):
                        lib = args[0]
                else:
                    continue
            else:
                if lib in libs_notfound:
                    continue
                if self.static:
                    mlog.warning('Static library {!r} not found for dependency {!r}, may not be statically linked'.format(lib[2:], self.name))
                libs_notfound.append(lib)
        elif lib.endswith('.la'):
            shared_libname = self.extract_libtool_shlib(lib)
            shared_lib = os.path.join(os.path.dirname(lib), shared_libname)
            if not os.path.exists(shared_lib):
                shared_lib = os.path.join(os.path.dirname(lib), '.libs', shared_libname)
            if not os.path.exists(shared_lib):
                raise DependencyException(f'Got a libtools specific "{lib}" dependenciesbut we could not compute the actual sharedlibrary path')
            self.is_libtool = True
            lib = shared_lib
            if lib in link_args:
                continue
        link_args.append(lib)
    if libs_notfound:
        link_args = ['-L' + lp for lp in prefix_libpaths] + link_args
    return (link_args, raw_link_args)