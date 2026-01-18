from __future__ import annotations
import functools
import typing as T
import os
import re
from ..environment import detect_cpu_family
from .base import DependencyMethods, detect_compiler, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import factory_methods
from .pkgconfig import PkgConfigDependency
class OpenMPIConfigToolDependency(_MPIConfigToolDependency):
    """Wrapper around OpenMPI mpicc and friends."""
    version_arg = '--showme:version'

    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any], language: T.Optional[str]=None):
        super().__init__(name, env, kwargs, language=language)
        if not self.is_found:
            return
        c_args = self.get_config_value(['--showme:compile'], 'compile_args')
        self.compile_args = self._filter_compile_args(c_args)
        l_args = self.get_config_value(['--showme:link'], 'link_args')
        self.link_args = self._filter_link_args(l_args)

    def _sanitize_version(self, out: str) -> str:
        v = re.search('\\d+.\\d+.\\d+', out)
        if v:
            return v.group(0)
        return out