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
def _find_source_script(self, name: str, prog: T.Union[str, mesonlib.File, build.Executable, ExternalProgram], args: T.List[str]) -> 'ExecutableSerialisation':
    largs: T.List[T.Union[str, build.Executable, ExternalProgram]] = []
    if isinstance(prog, (build.Executable, ExternalProgram)):
        FeatureNew.single_use(f'Passing executable/found program object to script parameter of {name}', '0.55.0', self.subproject, location=self.current_node)
        largs.append(prog)
    else:
        if isinstance(prog, mesonlib.File):
            FeatureNew.single_use(f'Passing file object to script parameter of {name}', '0.57.0', self.subproject, location=self.current_node)
        found = self.interpreter.find_program_impl([prog])
        largs.append(found)
    largs.extend(args)
    es = self.interpreter.backend.get_executable_serialisation(largs, verbose=True)
    es.subproject = self.interpreter.subproject
    return es