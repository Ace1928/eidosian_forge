from __future__ import annotations
from os import path
import shlex
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build
from .. import mesonlib
from .. import mlog
from ..interpreter.type_checking import CT_BUILD_BY_DEFAULT, CT_INPUT_KW, INSTALL_TAG_KW, OUTPUT_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, in_set_validator
from ..interpreterbase import FeatureNew, InvalidArguments
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, noPosargs, typed_kwargs, typed_pos_args
from ..programs import ExternalProgram
from ..scripts.gettext import read_linguas
class MergeFile(TypedDict):
    input: T.List[T.Union[str, build.BuildTarget, build.CustomTarget, build.CustomTargetIndex, build.ExtractedObjects, build.GeneratedList, ExternalProgram, mesonlib.File]]
    output: str
    build_by_default: bool
    install: bool
    install_dir: T.Optional[str]
    install_tag: T.Optional[str]
    args: T.List[str]
    data_dirs: T.List[str]
    po_dir: str
    type: Literal['xml', 'desktop']