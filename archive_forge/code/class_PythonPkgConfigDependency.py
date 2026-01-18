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
class PythonPkgConfigDependency(PkgConfigDependency, _PythonDependencyBase):

    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any], installation: 'BasicPythonExternalProgram', libpc: bool=False):
        if libpc:
            mlog.debug(f'Searching for {name!r} via pkgconfig lookup in LIBPC')
        else:
            mlog.debug(f'Searching for {name!r} via fallback pkgconfig lookup in default paths')
        PkgConfigDependency.__init__(self, name, environment, kwargs)
        _PythonDependencyBase.__init__(self, installation, kwargs.get('embed', False))
        if libpc and (not self.is_found):
            mlog.debug(f'"python-{self.version}" could not be found in LIBPC, this is likely due to a relocated python installation')
        if not self.link_libpython and mesonlib.version_compare(self.version, '< 3.8'):
            self.link_args = []