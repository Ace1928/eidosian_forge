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
class QtExtraFrameworkDependency(ExtraFrameworkDependency):

    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any], qvars: T.Dict[str, str], language: T.Optional[str]=None):
        super().__init__(name, env, kwargs, language=language)
        self.mod_name = name[2:]
        self.qt_extra_include_directory = qvars['QT_INSTALL_HEADERS']

    def get_compile_args(self, with_private_headers: bool=False, qt_version: str='0') -> T.List[str]:
        if self.found():
            mod_inc_dir = os.path.join(self.framework_path, 'Headers')
            args = ['-I' + mod_inc_dir]
            if with_private_headers:
                args += ['-I' + dirname for dirname in _qt_get_private_includes(mod_inc_dir, self.mod_name, qt_version)]
            if self.qt_extra_include_directory:
                args += ['-I' + self.qt_extra_include_directory]
            return args
        return []