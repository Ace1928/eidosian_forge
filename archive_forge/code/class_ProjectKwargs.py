from __future__ import annotations
import itertools
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build
from .. import mesonlib
from ..interpreter.type_checking import CT_INPUT_KW
from ..interpreterbase.decorators import KwargInfo, typed_kwargs, typed_pos_args
class ProjectKwargs(TypedDict):
    sources: T.List[T.Union[mesonlib.FileOrString, build.GeneratedTypes]]
    constraint_file: T.Union[mesonlib.FileOrString, build.GeneratedTypes]