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
@FeatureNew('add_devenv', '0.58.0')
@typed_kwargs('environment', ENV_METHOD_KW, ENV_SEPARATOR_KW.evolve(since='0.62.0'))
@typed_pos_args('add_devenv', (str, list, dict, mesonlib.EnvironmentVariables))
def add_devenv_method(self, args: T.Tuple[T.Union[str, list, dict, mesonlib.EnvironmentVariables]], kwargs: 'AddDevenvKW') -> None:
    env = args[0]
    msg = ENV_KW.validator(env)
    if msg:
        raise build.InvalidArguments(f'"add_devenv": {msg}')
    converted = env_convertor_with_method(env, kwargs['method'], kwargs['separator'])
    assert isinstance(converted, mesonlib.EnvironmentVariables)
    self.build.devenv.append(converted)