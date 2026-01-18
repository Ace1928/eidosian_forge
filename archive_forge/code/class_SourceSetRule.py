from __future__ import annotations
import typing as T
from . import ExtensionModule, ModuleObject, MutableModuleObject, ModuleInfo
from .. import build
from .. import dependencies
from .. import mesonlib
from ..interpreterbase import (
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import OrderedSet
class SourceSetRule(T.NamedTuple):
    keys: T.List[str]
    'Configuration keys that enable this rule if true'
    deps: T.List[dependencies.Dependency]
    'Dependencies that enable this rule if true'
    sources: T.List[T.Union[mesonlib.FileOrString, build.GeneratedTypes]]
    "Source files added when this rule's conditions are true"
    extra_deps: T.List[dependencies.Dependency]
    "Dependencies added when this rule's conditions are true, but\n       that do not make the condition false if they're absent."
    sourcesets: T.List[SourceSetImpl]
    "Other sourcesets added when this rule's conditions are true"
    if_false: T.List[T.Union[mesonlib.FileOrString, build.GeneratedTypes]]
    "Source files added when this rule's conditions are false"