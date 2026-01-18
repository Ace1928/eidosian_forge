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
def _parse_qrc_deps(self, state: 'ModuleState', rcc_file_: T.Union['FileOrString', build.CustomTarget, build.CustomTargetIndex, build.GeneratedList]) -> T.List[File]:
    result: T.List[File] = []
    inputs: T.Sequence['FileOrString'] = []
    if isinstance(rcc_file_, (str, File)):
        inputs = [rcc_file_]
    else:
        inputs = rcc_file_.get_outputs()
    for rcc_file in inputs:
        rcc_dirname, nodes = self._qrc_nodes(state, rcc_file)
        for resource_path in nodes:
            if os.path.isabs(resource_path):
                if resource_path.startswith(os.path.abspath(state.environment.build_dir)):
                    resource_relpath = os.path.relpath(resource_path, state.environment.build_dir)
                    result.append(File(is_built=True, subdir='', fname=resource_relpath))
                else:
                    result.append(File(is_built=False, subdir=state.subdir, fname=resource_path))
            else:
                path_from_rcc = os.path.normpath(os.path.join(rcc_dirname, resource_path))
                if path_from_rcc.startswith(state.environment.build_dir):
                    result.append(File(is_built=True, subdir=state.subdir, fname=resource_path))
                else:
                    result.append(File(is_built=False, subdir=state.subdir, fname=path_from_rcc))
    return result