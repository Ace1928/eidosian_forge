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
class PythonIntrospectionDict(TypedDict):
    install_paths: T.Dict[str, str]
    is_pypy: bool
    is_venv: bool
    link_libpython: bool
    sysconfig_paths: T.Dict[str, str]
    paths: T.Dict[str, str]
    platform: str
    suffix: str
    limited_api_suffix: str
    variables: T.Dict[str, str]
    version: str