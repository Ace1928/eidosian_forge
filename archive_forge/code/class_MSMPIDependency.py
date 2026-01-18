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
class MSMPIDependency(SystemDependency):
    """The Microsoft MPI."""

    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any], language: T.Optional[str]=None):
        super().__init__(name, env, kwargs, language=language)
        if language not in {'c', 'fortran', None}:
            self.is_found = False
            return
        if not self.env.machines[self.for_machine].is_windows():
            return
        incdir = os.environ.get('MSMPI_INC')
        arch = detect_cpu_family(self.env.coredata.compilers.host)
        libdir = None
        if arch == 'x86':
            libdir = os.environ.get('MSMPI_LIB32')
            post = 'x86'
        elif arch == 'x86_64':
            libdir = os.environ.get('MSMPI_LIB64')
            post = 'x64'
        if libdir is None or incdir is None:
            self.is_found = False
            return
        self.is_found = True
        self.link_args = ['-l' + os.path.join(libdir, 'msmpi')]
        self.compile_args = ['-I' + incdir, '-I' + os.path.join(incdir, post)]
        if self.language == 'fortran':
            self.link_args.append('-l' + os.path.join(libdir, 'msmpifec'))