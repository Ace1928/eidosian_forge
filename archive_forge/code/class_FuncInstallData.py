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
class FuncInstallData(TypedDict):
    install_dir: str
    sources: T.List[FileOrString]
    rename: T.List[str]
    install_mode: FileMode
    follow_symlinks: T.Optional[bool]