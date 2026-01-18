from __future__ import annotations
import re
import os, os.path, pathlib
import shutil
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleObject, ModuleInfo
from .. import build, mesonlib, mlog, dependencies
from ..cmake import TargetOptions, cmake_defines_to_args
from ..interpreter import SubprojectHolder
from ..interpreter.type_checking import REQUIRED_KW, INSTALL_DIR_KW, NoneType, in_set_validator
from ..interpreterbase import (
@typed_pos_args('subproject_options.add_cmake_defines', varargs=dict)
@noKwargs
def add_cmake_defines(self, state: ModuleState, args: T.Tuple[T.List[T.Dict[str, TYPE_var]]], kwargs: TYPE_kwargs) -> None:
    self.cmake_options += cmake_defines_to_args(args[0])