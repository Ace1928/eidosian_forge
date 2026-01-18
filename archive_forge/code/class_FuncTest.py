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
class FuncTest(_kwargs.BaseTest):
    dependencies: T.List[T.Union[Dependency, ExternalLibrary]]
    is_parallel: bool
    link_with: T.List[LibTypes]
    rust_args: T.List[str]