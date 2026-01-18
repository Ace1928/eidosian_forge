from __future__ import annotations
import os
import shutil
import typing as T
import xml.etree.ElementTree as ET
from . import ModuleReturnValue, ExtensionModule
from .. import build
from .. import coredata
from .. import mlog
from ..dependencies import find_external_dependency, Dependency, ExternalLibrary, InternalDependency
from ..mesonlib import MesonException, File, version_compare, Popen_safe
from ..interpreter import extract_required_kwarg
from ..interpreter.type_checking import INSTALL_DIR_KW, INSTALL_KW, NoneType
from ..interpreterbase import ContainerTypeInfo, FeatureDeprecated, KwargInfo, noPosargs, FeatureNew, typed_kwargs
from ..programs import NonExistingExternalProgram
class CompileTranslationsKwArgs(TypedDict):
    build_by_default: bool
    install: bool
    install_dir: T.Optional[str]
    method: str
    qresource: T.Optional[str]
    rcc_extra_arguments: T.List[str]
    ts_files: T.List[T.Union[str, File, build.CustomTarget, build.CustomTargetIndex, build.GeneratedList]]