from __future__ import annotations
import os
import typing as T
from .. import mesonlib
from .. import dependencies
from .. import build
from .. import mlog, coredata
from ..mesonlib import MachineChoice, OptionKey
from ..programs import OverrideProgram, ExternalProgram
from ..interpreter.type_checking import ENV_KW, ENV_METHOD_KW, ENV_SEPARATOR_KW, env_convertor_with_method
from ..interpreterbase import (MesonInterpreterObject, FeatureNew, FeatureDeprecated,
from .primitives import MesonVersionString
from .type_checking import NATIVE_KW, NoneType
@typed_pos_args('meson.add_dist_script', (str, mesonlib.File, ExternalProgram), varargs=(str, mesonlib.File, ExternalProgram))
@noKwargs
@FeatureNew('meson.add_dist_script', '0.48.0')
def add_dist_script_method(self, args: T.Tuple[T.Union[str, mesonlib.File, ExternalProgram], T.List[T.Union[str, mesonlib.File, ExternalProgram]]], kwargs: 'TYPE_kwargs') -> None:
    if args[1]:
        FeatureNew.single_use('Calling "add_dist_script" with multiple arguments', '0.49.0', self.interpreter.subproject, location=self.current_node)
    if self.interpreter.subproject != '':
        FeatureNew.single_use('Calling "add_dist_script" in a subproject', '0.58.0', self.interpreter.subproject, location=self.current_node)
    script_args = self._process_script_args('add_dist_script', args[1])
    script = self._find_source_script('add_dist_script', args[0], script_args)
    self.build.dist_scripts.append(script)