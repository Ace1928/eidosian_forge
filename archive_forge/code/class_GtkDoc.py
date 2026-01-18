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
class GtkDoc(TypedDict):
    src_dir: T.List[T.Union[str, build.IncludeDirs]]
    main_sgml: str
    main_xml: str
    module_version: str
    namespace: str
    mode: Literal['xml', 'smgl', 'auto', 'none']
    html_args: T.List[str]
    scan_args: T.List[str]
    scanobjs_args: T.List[str]
    fixxref_args: T.List[str]
    mkdb_args: T.List[str]
    content_files: T.List[T.Union[build.GeneratedTypes, FileOrString]]
    ignore_headers: T.List[str]
    install_dir: T.List[str]
    check: bool
    install: bool
    gobject_typesfile: T.List[FileOrString]
    html_assets: T.List[FileOrString]
    expand_content_files: T.List[FileOrString]
    c_args: T.List[str]
    include_directories: T.List[T.Union[str, build.IncludeDirs]]
    dependencies: T.List[T.Union[Dependency, build.SharedLibrary, build.StaticLibrary]]