from __future__ import annotations
import os
import shlex
import subprocess
import copy
import textwrap
from pathlib import Path, PurePath
from .. import mesonlib
from .. import coredata
from .. import build
from .. import mlog
from ..modules import ModuleReturnValue, ModuleObject, ModuleState, ExtensionModule
from ..backend.backends import TestProtocol
from ..interpreterbase import (
from ..interpreter.type_checking import NoneType, ENV_KW, ENV_SEPARATOR_KW, PKGCONFIG_DEFINE_KW
from ..dependencies import Dependency, ExternalLibrary, InternalDependency
from ..programs import ExternalProgram
from ..mesonlib import HoldableObject, OptionKey, listify, Popen_safe
import typing as T
class _CustomTargetHolder(ObjectHolder[_CT]):

    def __init__(self, target: _CT, interp: 'Interpreter'):
        super().__init__(target, interp)
        self.methods.update({'full_path': self.full_path_method, 'to_list': self.to_list_method})
        self.operators.update({MesonOperator.INDEX: self.op_index})

    def __repr__(self) -> str:
        r = '<{} {}: {}>'
        h = self.held_object
        return r.format(self.__class__.__name__, h.get_id(), h.command)

    @noPosargs
    @noKwargs
    def full_path_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self.interpreter.backend.get_target_filename_abs(self.held_object)

    @FeatureNew('custom_target.to_list', '0.54.0')
    @noPosargs
    @noKwargs
    def to_list_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> T.List[build.CustomTargetIndex]:
        result = []
        for i in self.held_object:
            result.append(i)
        return result

    @noKwargs
    @typed_operator(MesonOperator.INDEX, int)
    def op_index(self, other: int) -> build.CustomTargetIndex:
        try:
            return self.held_object[other]
        except IndexError:
            raise InvalidArguments(f'Index {other} out of bounds of custom target {self.held_object.name} output of size {len(self.held_object)}.')