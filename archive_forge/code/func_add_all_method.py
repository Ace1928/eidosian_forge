from __future__ import annotations
import typing as T
from . import ExtensionModule, ModuleObject, MutableModuleObject, ModuleInfo
from .. import build
from .. import dependencies
from .. import mesonlib
from ..interpreterbase import (
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import OrderedSet
@typed_pos_args('sourceset.add_all', varargs=SourceSet)
@typed_kwargs('sourceset.add_all', _WHEN_KW, KwargInfo('if_true', ContainerTypeInfo(list, SourceSet), listify=True, default=[]))
def add_all_method(self, state: ModuleState, args: T.Tuple[T.List[SourceSetImpl]], kwargs: AddAllKw) -> None:
    if self.frozen:
        raise InvalidCode("Tried to use 'add_all' after querying the source set")
    when = kwargs['when']
    if_true = kwargs['if_true']
    if not when and (not if_true):
        if_true = args[0]
    elif args[0]:
        raise InterpreterException('add_all called with both positional and keyword arguments')
    keys, dependencies = self.check_conditions(when)
    for s in if_true:
        s.frozen = True
    self.rules.append(SourceSetRule(keys, dependencies, [], [], if_true, []))