from __future__ import annotations
import enum
import os
import re
import typing as T
from . import ExtensionModule, ModuleInfo
from . import ModuleReturnValue
from .. import mesonlib, build
from .. import mlog
from ..interpreter.type_checking import DEPEND_FILES_KW, DEPENDS_KW, INCLUDE_DIRECTORIES
from ..interpreterbase.decorators import ContainerTypeInfo, FeatureNew, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import MachineChoice, MesonException
from ..programs import ExternalProgram
class ResourceCompilerType(enum.Enum):
    windres = 1
    rc = 2
    wrc = 3