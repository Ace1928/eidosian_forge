from __future__ import annotations
import functools, json, os, textwrap
from pathlib import Path
import typing as T
from .. import mesonlib, mlog
from .base import process_method_kw, DependencyException, DependencyMethods, DependencyTypeName, ExternalDependency, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
from .framework import ExtraFrameworkDependency
from .pkgconfig import PkgConfigDependency
from ..environment import detect_cpu_family
from ..programs import ExternalProgram
class PythonSystemDependency(SystemDependency, _PythonDependencyBase):

    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any], installation: 'BasicPythonExternalProgram'):
        SystemDependency.__init__(self, name, environment, kwargs)
        _PythonDependencyBase.__init__(self, installation, kwargs.get('embed', False))
        if self.link_libpython:
            if mesonlib.is_windows():
                self.find_libpy_windows(environment, limited_api=False)
            else:
                self.find_libpy(environment)
        else:
            self.is_found = True
        inc_paths = mesonlib.OrderedSet([self.variables.get('INCLUDEPY'), self.paths.get('include'), self.paths.get('platinclude')])
        self.compile_args += ['-I' + path for path in inc_paths if path]
        if mesonlib.is_windows() and self.get_windows_python_arch().endswith('64') and mesonlib.version_compare(self.version, '<3.12'):
            self.compile_args += ['-DMS_WIN64=']
        if not self.clib_compiler.has_header('Python.h', '', environment, extra_args=self.compile_args):
            self.is_found = False

    def find_libpy(self, environment: 'Environment') -> None:
        if self.is_pypy:
            if self.major_version == 3:
                libname = 'pypy3-c'
            else:
                libname = 'pypy-c'
            libdir = os.path.join(self.variables.get('base'), 'bin')
            libdirs = [libdir]
        else:
            libname = f'python{self.version}'
            if 'DEBUG_EXT' in self.variables:
                libname += self.variables['DEBUG_EXT']
            if 'ABIFLAGS' in self.variables:
                libname += self.variables['ABIFLAGS']
            libdirs = []
        largs = self.clib_compiler.find_library(libname, environment, libdirs)
        if largs is not None:
            self.link_args = largs
            self.is_found = True

    def get_windows_python_arch(self) -> str:
        if self.platform.startswith('mingw'):
            if 'x86_64' in self.platform:
                return 'x86_64'
            elif 'i686' in self.platform:
                return 'x86'
            elif 'aarch64' in self.platform:
                return 'aarch64'
            else:
                raise DependencyException(f'MinGW Python built with unknown platform {self.platform!r}, please file a bug')
        elif self.platform == 'win32':
            return 'x86'
        elif self.platform in {'win64', 'win-amd64'}:
            return 'x86_64'
        elif self.platform in {'win-arm64'}:
            return 'aarch64'
        raise DependencyException('Unknown Windows Python platform {self.platform!r}')

    def get_windows_link_args(self, limited_api: bool) -> T.Optional[T.List[str]]:
        if self.platform.startswith('win'):
            vernum = self.variables.get('py_version_nodot')
            verdot = self.variables.get('py_version_short')
            imp_lower = self.variables.get('implementation_lower', 'python')
            if self.static:
                libpath = Path('libs') / f'libpython{vernum}.a'
            else:
                comp = self.get_compiler()
                if comp.id == 'gcc':
                    if imp_lower == 'pypy' and verdot == '3.8':
                        libpath = Path('libpypy3-c.dll')
                    elif imp_lower == 'pypy':
                        libpath = Path(f'libpypy{verdot}-c.dll')
                    else:
                        libpath = Path(f'python{vernum}.dll')
                else:
                    if limited_api:
                        vernum = vernum[0]
                    libpath = Path('libs') / f'python{vernum}.lib'
                    buildtype = self.env.coredata.get_option(mesonlib.OptionKey('buildtype'))
                    assert isinstance(buildtype, str)
                    debug = self.env.coredata.get_option(mesonlib.OptionKey('debug'))
                    is_debug_build = debug or buildtype == 'debug'
                    vscrt_debug = False
                    if mesonlib.OptionKey('b_vscrt') in self.env.coredata.options:
                        vscrt = self.env.coredata.options[mesonlib.OptionKey('b_vscrt')].value
                        if vscrt in {'mdd', 'mtd', 'from_buildtype', 'static_from_buildtype'}:
                            vscrt_debug = True
                    if is_debug_build and vscrt_debug and (not self.variables.get('Py_DEBUG')):
                        mlog.warning(textwrap.dedent('                            Using a debug build type with MSVC or an MSVC-compatible compiler\n                            when the Python interpreter is not also a debug build will almost\n                            certainly result in a failed build. Prefer using a release build\n                            type or a debug Python interpreter.\n                            '))
            lib = Path(self.variables.get('base_prefix')) / libpath
        elif self.platform.startswith('mingw'):
            if self.static:
                libname = self.variables.get('LIBRARY')
            else:
                libname = self.variables.get('LDLIBRARY')
            lib = Path(self.variables.get('LIBDIR')) / libname
        else:
            raise mesonlib.MesonBugException("On a Windows path, but the OS doesn't appear to be Windows or MinGW.")
        if not lib.exists():
            mlog.log('Could not find Python3 library {!r}'.format(str(lib)))
            return None
        return [str(lib)]

    def find_libpy_windows(self, env: 'Environment', limited_api: bool=False) -> None:
        """
        Find python3 libraries on Windows and also verify that the arch matches
        what we are building for.
        """
        try:
            pyarch = self.get_windows_python_arch()
        except DependencyException as e:
            mlog.log(str(e))
            self.is_found = False
            return
        arch = detect_cpu_family(env.coredata.compilers.host)
        if arch != pyarch:
            mlog.log('Need', mlog.bold(self.name), f'for {arch}, but found {pyarch}')
            self.is_found = False
            return
        largs = self.get_windows_link_args(limited_api)
        if largs is None:
            self.is_found = False
            return
        self.link_args = largs
        self.is_found = True

    @staticmethod
    def log_tried() -> str:
        return 'sysconfig'