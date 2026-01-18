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
def add_build_def_file(self, f: mesonlib.FileOrString) -> None:
    if isinstance(f, mesonlib.File):
        if f.is_built:
            return
        f = os.path.normpath(f.relative_name())
    elif os.path.isfile(f) and (not f.startswith('/dev/')):
        srcdir = Path(self.environment.get_source_dir())
        builddir = Path(self.environment.get_build_dir())
        try:
            f_ = Path(f).resolve()
        except OSError:
            f_ = Path(f)
            s = f_.stat()
            if hasattr(s, 'st_file_attributes') and s.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT != 0 and (s.st_reparse_tag == stat.IO_REPARSE_TAG_APPEXECLINK):
                f_ = f_.parent.resolve() / f_.name
            else:
                raise
        if builddir in f_.parents:
            return
        if srcdir in f_.parents:
            f_ = f_.relative_to(srcdir)
        f = str(f_)
    else:
        return
    if f not in self.build_def_files:
        self.build_def_files.add(f)