from __future__ import annotations
import typing as T
from . import ExtensionModule, ModuleObject, MutableModuleObject, ModuleInfo
from .. import build
from .. import dependencies
from .. import mesonlib
from ..interpreterbase import (
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import OrderedSet
class SourceFiles(T.NamedTuple):
    sources: OrderedSet[T.Union[mesonlib.FileOrString, build.GeneratedTypes]]
    deps: OrderedSet[dependencies.Dependency]