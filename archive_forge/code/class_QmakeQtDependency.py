from __future__ import annotations
import abc
import re
import os
import typing as T
from .base import DependencyException, DependencyMethods
from .configtool import ConfigToolDependency
from .detect import packages
from .framework import ExtraFrameworkDependency
from .pkgconfig import PkgConfigDependency
from .factory import DependencyFactory
from .. import mlog
from .. import mesonlib
class QmakeQtDependency(_QtBase, ConfigToolDependency, metaclass=abc.ABCMeta):
    """Find Qt using Qmake as a config-tool."""
    version: str
    version_arg = '-v'

    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any]):
        _QtBase.__init__(self, name, kwargs)
        self.tool_name = f'qmake{self.qtver}'
        self.tools = [f'qmake{self.qtver}', f'qmake-{self.name}', 'qmake']
        kwargs = kwargs.copy()
        _vers = mesonlib.listify(kwargs.get('version', []))
        _vers.extend([f'>= {self.qtver}', f'< {int(self.qtver) + 1}'])
        kwargs['version'] = _vers
        ConfigToolDependency.__init__(self, name, env, kwargs)
        if not self.found():
            return
        stdo = self.get_config_value(['-query'], 'args')
        qvars: T.Dict[str, str] = {}
        for line in stdo:
            line = line.strip()
            if line == '':
                continue
            k, v = line.split(':', 1)
            qvars[k] = v
        xspec = qvars.get('QMAKE_XSPEC', '')
        if self.env.machines.host.is_darwin() and (not any((s in xspec for s in ['ios', 'tvos']))):
            mlog.debug('Building for macOS, looking for framework')
            self._framework_detect(qvars, self.requested_modules, kwargs)
            if self.is_found:
                return
            else:
                mlog.debug("Building for macOS, couldn't find framework, falling back to library search")
        incdir = qvars['QT_INSTALL_HEADERS']
        self.compile_args.append('-I' + incdir)
        libdir = qvars['QT_INSTALL_LIBS']
        self.bindir = get_qmake_host_bins(qvars)
        self.libexecdir = get_qmake_host_libexecs(qvars)
        is_debug = self.env.coredata.get_option(mesonlib.OptionKey('buildtype')) == 'debug'
        if mesonlib.OptionKey('b_vscrt') in self.env.coredata.options:
            if self.env.coredata.options[mesonlib.OptionKey('b_vscrt')].value in {'mdd', 'mtd'}:
                is_debug = True
        modules_lib_suffix = _get_modules_lib_suffix(self.version, self.env.machines[self.for_machine], is_debug)
        for module in self.requested_modules:
            mincdir = os.path.join(incdir, 'Qt' + module)
            self.compile_args.append('-I' + mincdir)
            if module == 'QuickTest':
                define_base = 'QMLTEST'
            elif module == 'Test':
                define_base = 'TESTLIB'
            else:
                define_base = module.upper()
            self.compile_args.append(f'-DQT_{define_base}_LIB')
            if self.private_headers:
                priv_inc = self.get_private_includes(mincdir, module)
                for directory in priv_inc:
                    self.compile_args.append('-I' + directory)
            libfiles = self.clib_compiler.find_library(self.qtpkgname + module + modules_lib_suffix, self.env, mesonlib.listify(libdir))
            if libfiles:
                libfile = libfiles[0]
            else:
                mlog.log('Could not find:', module, self.qtpkgname + module + modules_lib_suffix, 'in', libdir)
                self.is_found = False
                break
            self.link_args.append(libfile)
        if self.env.machines[self.for_machine].is_windows() and self.qtmain:
            if not self._link_with_qt_winmain(is_debug, libdir):
                self.is_found = False

    def _sanitize_version(self, version: str) -> str:
        m = re.search(f'({self.qtver}(\\.\\d+)+)', version)
        if m:
            return m.group(0).rstrip('.')
        return version

    def get_variable_args(self, variable_name: str) -> T.List[str]:
        return ['-query', f'{variable_name}']

    @abc.abstractmethod
    def get_private_includes(self, mod_inc_dir: str, module: str) -> T.List[str]:
        pass

    def _framework_detect(self, qvars: T.Dict[str, str], modules: T.List[str], kwargs: T.Dict[str, T.Any]) -> None:
        libdir = qvars['QT_INSTALL_LIBS']
        fw_kwargs = kwargs.copy()
        fw_kwargs.pop('method', None)
        fw_kwargs['paths'] = [libdir]
        for m in modules:
            fname = 'Qt' + m
            mlog.debug('Looking for qt framework ' + fname)
            fwdep = QtExtraFrameworkDependency(fname, self.env, fw_kwargs, qvars, language=self.language)
            if fwdep.found():
                self.compile_args.append('-F' + libdir)
                self.compile_args += fwdep.get_compile_args(with_private_headers=self.private_headers, qt_version=self.version)
                self.link_args += fwdep.get_link_args()
            else:
                self.is_found = False
                break
        else:
            self.is_found = True
            self.bindir = get_qmake_host_bins(qvars)
            self.libexecdir = get_qmake_host_libexecs(qvars)

    def log_info(self) -> str:
        return 'qmake'