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
def _args_to_info(self, args: T.List[str]) -> T.Dict[str, str]:
    if len(args) != 1:
        raise InterpreterException('Exactly one argument is required.')
    tgt = args[0]
    res = self.cm_interpreter.target_info(tgt)
    if res is None:
        raise InterpreterException(f'The CMake target {tgt} does not exist\n' + '  Use the following command in your meson.build to list all available targets:\n\n' + "    message('CMake targets:\\n - ' + '\\n - '.join(<cmake_subproject>.target_list()))")
    assert all((x in res for x in ['inc', 'src', 'dep', 'tgt', 'func']))
    return res