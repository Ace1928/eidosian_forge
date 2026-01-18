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
@staticmethod
def _find_tool(state: 'ModuleState', tool: str) -> 'ToolType':
    tool_map = {'gio-querymodules': 'gio-2.0', 'glib-compile-schemas': 'gio-2.0', 'glib-compile-resources': 'gio-2.0', 'gdbus-codegen': 'gio-2.0', 'glib-genmarshal': 'glib-2.0', 'glib-mkenums': 'glib-2.0', 'g-ir-scanner': 'gobject-introspection-1.0', 'g-ir-compiler': 'gobject-introspection-1.0'}
    depname = tool_map[tool]
    varname = tool.replace('-', '_')
    return state.find_tool(tool, depname, varname)