from __future__ import annotations
import itertools
import os
import typing as T
from mesonbuild.interpreterbase.decorators import FeatureNew
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import mesonlib, mlog
from ..build import (BothLibraries, BuildTarget, CustomTargetIndex, Executable, ExtractedObjects, GeneratedList,
from ..compilers.compilers import are_asserts_disabled, lang_suffixes
from ..interpreter.type_checking import (
from ..interpreterbase import ContainerTypeInfo, InterpreterException, KwargInfo, typed_kwargs, typed_pos_args, noPosargs, permittedKwargs
from ..mesonlib import File
from ..programs import ExternalProgram
class FuncBindgen(TypedDict):
    args: T.List[str]
    c_args: T.List[str]
    include_directories: T.List[IncludeDirs]
    input: T.List[SourceInputs]
    output: str
    output_inline_wrapper: str
    dependencies: T.List[T.Union[Dependency, ExternalLibrary]]
    language: T.Optional[Literal['c', 'cpp']]
    bindgen_version: T.List[str]