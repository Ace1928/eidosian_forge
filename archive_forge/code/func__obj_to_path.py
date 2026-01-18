from __future__ import annotations
from pathlib import Path, PurePath, PureWindowsPath
import hashlib
import os
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import mlog
from ..build import BuildTarget, CustomTarget, CustomTargetIndex, InvalidArguments
from ..interpreter.type_checking import INSTALL_KW, INSTALL_MODE_KW, INSTALL_TAG_KW, NoneType
from ..interpreterbase import FeatureNew, KwargInfo, typed_kwargs, typed_pos_args, noKwargs
from ..mesonlib import File, MesonException, has_path_sep, path_is_in_root, relpath
@staticmethod
def _obj_to_path(feature_new_prefix: str, obj: T.Union[FileOrString, BuildTargetTypes], state: ModuleState) -> PurePath:
    if isinstance(obj, str):
        return PurePath(obj)
    if isinstance(obj, File):
        FeatureNew(f'{feature_new_prefix} with file', '0.59.0').use(state.subproject, location=state.current_node)
        return PurePath(str(obj))
    FeatureNew(f'{feature_new_prefix} with build_tgt, custom_tgt, and custom_idx', '1.4.0').use(state.subproject, location=state.current_node)
    return PurePath(state.backend.get_target_filename(obj))