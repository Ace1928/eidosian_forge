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
def _get_gir_dep(self, state: 'ModuleState') -> T.Tuple[Dependency, T.Union[Executable, 'ExternalProgram', 'OverrideProgram'], T.Union[Executable, 'ExternalProgram', 'OverrideProgram']]:
    if not self.gir_dep:
        self.gir_dep = state.dependency('gobject-introspection-1.0')
        self.giscanner = self._find_tool(state, 'g-ir-scanner')
        self.gicompiler = self._find_tool(state, 'g-ir-compiler')
    return (self.gir_dep, self.giscanner, self.gicompiler)