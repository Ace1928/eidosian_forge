from __future__ import annotations
import typing as T
from typing_extensions import TypedDict, Literal, Protocol, NotRequired
from .. import build
from .. import coredata
from ..compilers import Compiler
from ..dependencies.base import Dependency
from ..mesonlib import EnvironmentVariables, MachineChoice, File, FileMode, FileOrString, OptionKey
from ..modules.cmake import CMakeSubprojectOptions
from ..programs import ExternalProgram
from .type_checking import PkgConfigDefineType, SourcesVarargsType
class FuncGenerator(TypedDict):
    """Keyword rguments for the generator function."""
    arguments: T.List[str]
    output: T.List[str]
    depfile: T.Optional[str]
    capture: bool
    depends: T.List[T.Union[build.BuildTarget, build.CustomTarget]]