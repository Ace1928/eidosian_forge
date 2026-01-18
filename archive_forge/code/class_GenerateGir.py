from __future__ import annotations
import copy
import itertools
import functools
import os
import subprocess
import textwrap
import typing as T
from . import (
from .. import build
from .. import interpreter
from .. import mesonlib
from .. import mlog
from ..build import CustomTarget, CustomTargetIndex, Executable, GeneratedList, InvalidArguments
from ..dependencies import Dependency, InternalDependency
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import DEPENDS_KW, DEPEND_FILES_KW, ENV_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, DEPENDENCY_SOURCES_KW, in_set_validator
from ..interpreterbase import noPosargs, noKwargs, FeatureNew, FeatureDeprecated
from ..interpreterbase import typed_kwargs, KwargInfo, ContainerTypeInfo
from ..interpreterbase.decorators import typed_pos_args
from ..mesonlib import (
from ..programs import OverrideProgram
from ..scripts.gettext import read_linguas
class GenerateGir(TypedDict):
    build_by_default: bool
    dependencies: T.List[Dependency]
    export_packages: T.List[str]
    extra_args: T.List[str]
    fatal_warnings: bool
    header: T.List[str]
    identifier_prefix: T.List[str]
    include_directories: T.List[T.Union[build.IncludeDirs, str]]
    includes: T.List[T.Union[str, GirTarget]]
    install: bool
    install_dir_gir: T.Optional[str]
    install_dir_typelib: T.Optional[str]
    link_with: T.List[T.Union[build.SharedLibrary, build.StaticLibrary]]
    namespace: str
    nsversion: str
    sources: T.List[T.Union[FileOrString, build.GeneratedTypes]]
    symbol_prefix: T.List[str]