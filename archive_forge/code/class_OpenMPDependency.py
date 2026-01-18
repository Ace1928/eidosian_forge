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
class OpenMPDependency(SystemDependency):
    VERSIONS = {'202111': '5.2', '202011': '5.1', '201811': '5.0', '201611': '5.0-revision1', '201511': '4.5', '201307': '4.0', '201107': '3.1', '200805': '3.0', '200505': '2.5', '200203': '2.0', '199810': '1.0'}

    def __init__(self, environment: 'Environment', kwargs: T.Dict[str, T.Any]) -> None:
        language = kwargs.get('language')
        super().__init__('openmp', environment, kwargs, language=language)
        self.is_found = False
        if self.clib_compiler.get_id() == 'nagfor':
            self.version = '3.1'
            self.is_found = True
            self.compile_args = self.link_args = self.clib_compiler.openmp_flags()
            return
        if self.clib_compiler.get_id() == 'pgi':
            self.version = '3.1'
            self.is_found = True
            self.compile_args = self.link_args = self.clib_compiler.openmp_flags()
            return
        try:
            openmp_date = self.clib_compiler.get_define('_OPENMP', '', self.env, self.clib_compiler.openmp_flags(), [self], disable_cache=True)[0]
        except mesonlib.EnvironmentException as e:
            mlog.debug('OpenMP support not available in the compiler')
            mlog.debug(e)
            openmp_date = None
        if openmp_date:
            try:
                self.version = self.VERSIONS[openmp_date]
            except KeyError:
                mlog.debug(f'Could not find an OpenMP version matching {openmp_date}')
                if openmp_date == '_OPENMP':
                    mlog.debug("This can be caused by flags such as gcc's `-fdirectives-only`, which affect preprocessor behavior.")
                return
            if self.clib_compiler.get_id() == 'clang-cl':
                clangcl_openmp_link_args = self.clib_compiler.find_library('libomp', self.env, [])
                if not clangcl_openmp_link_args:
                    mlog.log(mlog.yellow('WARNING:'), 'OpenMP found but libomp for clang-cl missing.')
                    return
                self.link_args.extend(clangcl_openmp_link_args)
            header_names = ('omp.h', 'omp_lib.h')
            for name in header_names:
                if self.clib_compiler.has_header(name, '', self.env, dependencies=[self], disable_cache=True)[0]:
                    self.is_found = True
                    self.compile_args.extend(self.clib_compiler.openmp_flags())
                    self.link_args.extend(self.clib_compiler.openmp_link_flags())
                    break
            if not self.is_found:
                mlog.log(mlog.yellow('WARNING:'), 'OpenMP found but omp.h missing.')