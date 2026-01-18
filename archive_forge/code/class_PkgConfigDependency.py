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
class PkgConfigDependency(ExternalDependency):

    def __init__(self, name: str, environment: Environment, kwargs: T.Dict[str, T.Any], language: T.Optional[str]=None) -> None:
        super().__init__(DependencyTypeName('pkgconfig'), environment, kwargs, language=language)
        self.name = name
        self.is_libtool = False
        pkgconfig = PkgConfigInterface.instance(self.env, self.for_machine, self.silent)
        if not pkgconfig:
            msg = f'Pkg-config for machine {self.for_machine} not found. Giving up.'
            if self.required:
                raise DependencyException(msg)
            mlog.debug(msg)
            return
        self.pkgconfig = pkgconfig
        version = self.pkgconfig.version(name)
        if version is None:
            return
        self.version = version
        self.is_found = True
        try:
            self._set_cargs()
            self._set_libs()
        except DependencyException as e:
            mlog.debug(f"Pkg-config error with '{name}': {e}")
            if self.required:
                raise
            else:
                self.compile_args = []
                self.link_args = []
                self.is_found = False
                self.reason = e

    def __repr__(self) -> str:
        s = '<{0} {1}: {2} {3}>'
        return s.format(self.__class__.__name__, self.name, self.is_found, self.version_reqs)

    def _convert_mingw_paths(self, args: ImmutableListProtocol[str]) -> T.List[str]:
        """
        Both MSVC and native Python on Windows cannot handle MinGW-esque /c/foo
        paths so convert them to C:/foo. We cannot resolve other paths starting
        with / like /home/foo so leave them as-is so that the user gets an
        error/warning from the compiler/linker.
        """
        if not self.env.machines.build.is_windows():
            return args.copy()
        converted = []
        for arg in args:
            pargs: T.Tuple[str, ...] = tuple()
            if arg.startswith('-L/'):
                pargs = PurePath(arg[2:]).parts
                tmpl = '-L{}:/{}'
            elif arg.startswith('-I/'):
                pargs = PurePath(arg[2:]).parts
                tmpl = '-I{}:/{}'
            elif arg.startswith('/'):
                pargs = PurePath(arg).parts
                tmpl = '{}:/{}'
            elif arg.startswith(('-L', '-I')) or (len(arg) > 2 and arg[1] == ':'):
                arg = arg.replace('\\ ', ' ')
            if len(pargs) > 1 and len(pargs[1]) == 1:
                arg = tmpl.format(pargs[1], '/'.join(pargs[2:]))
            converted.append(arg)
        return converted

    def _set_cargs(self) -> None:
        allow_system = False
        if self.language == 'fortran':
            allow_system = True
        cflags = self.pkgconfig.cflags(self.name, allow_system)
        self.compile_args = self._convert_mingw_paths(cflags)

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

    def _set_libs(self) -> None:
        libs = self.pkgconfig.libs(self.name, self.static, allow_system=True)
        raw_libs = self.pkgconfig.libs(self.name, self.static, allow_system=False)
        self.link_args, self.raw_link_args = self._search_libs(libs, raw_libs)

    def extract_field(self, la_file: str, fieldname: str) -> T.Optional[str]:
        with open(la_file, encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split('=')
                if arr[0] == fieldname:
                    return arr[1][1:-1]
        return None

    def extract_dlname_field(self, la_file: str) -> T.Optional[str]:
        return self.extract_field(la_file, 'dlname')

    def extract_libdir_field(self, la_file: str) -> T.Optional[str]:
        return self.extract_field(la_file, 'libdir')

    def extract_libtool_shlib(self, la_file: str) -> T.Optional[str]:
        """
        Returns the path to the shared library
        corresponding to this .la file
        """
        dlname = self.extract_dlname_field(la_file)
        if dlname is None:
            return None
        if self.env.machines[self.for_machine].is_darwin():
            dlbasename = os.path.basename(dlname)
            libdir = self.extract_libdir_field(la_file)
            if libdir is None:
                return dlbasename
            return os.path.join(libdir, dlbasename)
        return os.path.basename(dlname)

    @staticmethod
    def log_tried() -> str:
        return 'pkgconfig'

    def get_variable(self, *, cmake: T.Optional[str]=None, pkgconfig: T.Optional[str]=None, configtool: T.Optional[str]=None, internal: T.Optional[str]=None, default_value: T.Optional[str]=None, pkgconfig_define: PkgConfigDefineType=None) -> str:
        if pkgconfig:
            try:
                variable = self.pkgconfig.variable(self.name, pkgconfig, pkgconfig_define)
                if variable is not None:
                    return variable
            except DependencyException:
                pass
        if default_value is not None:
            return default_value
        raise DependencyException(f'Could not get pkg-config variable and no default provided for {self!r}')