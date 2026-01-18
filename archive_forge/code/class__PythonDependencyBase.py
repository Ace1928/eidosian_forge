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
class _PythonDependencyBase(_Base):

    def __init__(self, python_holder: 'BasicPythonExternalProgram', embed: bool):
        self.embed = embed
        self.version: str = python_holder.info['version']
        self.platform = python_holder.info['platform']
        self.variables = python_holder.info['variables']
        self.paths = python_holder.info['paths']
        self.is_pypy = python_holder.info['is_pypy']
        self.link_libpython = python_holder.info['link_libpython'] or embed
        self.info: T.Optional[T.Dict[str, str]] = None
        if mesonlib.version_compare(self.version, '>= 3.0'):
            self.major_version = 3
        else:
            self.major_version = 2