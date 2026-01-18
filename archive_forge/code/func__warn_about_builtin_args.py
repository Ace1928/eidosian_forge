from __future__ import annotations
from .. import mparser
from .. import environment
from .. import coredata
from .. import dependencies
from .. import mlog
from .. import build
from .. import optinterpreter
from .. import compilers
from .. import envconfig
from ..wrap import wrap, WrapMode
from .. import mesonlib
from ..mesonlib import (EnvironmentVariables, ExecutableSerialisation, MesonBugException, MesonException, HoldableObject,
from ..programs import ExternalProgram, NonExistingExternalProgram
from ..dependencies import Dependency
from ..depfile import DepFile
from ..interpreterbase import ContainerTypeInfo, InterpreterBase, KwargInfo, typed_kwargs, typed_pos_args
from ..interpreterbase import noPosargs, noKwargs, permittedKwargs, noArgsFlattening, noSecondLevelHolderResolving, unholder_return
from ..interpreterbase import InterpreterException, InvalidArguments, InvalidCode, SubdirDoneRequest
from ..interpreterbase import Disabler, disablerIfNotFound
from ..interpreterbase import FeatureNew, FeatureDeprecated, FeatureBroken, FeatureNewKwargs
from ..interpreterbase import ObjectHolder, ContextManagerObject
from ..interpreterbase import stringifyUserArguments
from ..modules import ExtensionModule, ModuleObject, MutableModuleObject, NewExtensionModule, NotFoundExtensionModule
from ..optinterpreter import optname_regex
from . import interpreterobjects as OBJ
from . import compiler as compilerOBJ
from .mesonmain import MesonMain
from .dependencyfallbacks import DependencyFallbacksHolder
from .interpreterobjects import (
from .type_checking import (
from . import primitives as P_OBJ
from pathlib import Path
from enum import Enum
import os
import shutil
import uuid
import re
import stat
import collections
import typing as T
import textwrap
import importlib
import copy
def _warn_about_builtin_args(self, args: T.List[str]) -> None:
    warnargs = ('/W1', '/W2', '/W3', '/W4', '/Wall', '-Wall', '-Wextra')
    optargs = ('-O0', '-O2', '-O3', '-Os', '-Oz', '/O1', '/O2', '/Os')
    for arg in args:
        if arg in warnargs:
            mlog.warning(f'Consider using the built-in warning_level option instead of using "{arg}".', location=self.current_node)
        elif arg in optargs:
            mlog.warning(f'Consider using the built-in optimization level instead of using "{arg}".', location=self.current_node)
        elif arg == '-Werror':
            mlog.warning(f'Consider using the built-in werror option instead of using "{arg}".', location=self.current_node)
        elif arg == '-g':
            mlog.warning(f'Consider using the built-in debug option instead of using "{arg}".', location=self.current_node)
        elif arg in {'-fsanitize', '/fsanitize'} or arg.startswith(('-fsanitize=', '/fsanitize=')):
            mlog.warning(f'Consider using the built-in option for sanitizers instead of using "{arg}".', location=self.current_node)
        elif arg.startswith('-std=') or arg.startswith('/std:'):
            mlog.warning(f'Consider using the built-in option for language standard version instead of using "{arg}".', location=self.current_node)