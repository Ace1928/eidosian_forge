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
class CMakeSubprojectOptions(ModuleObject):

    def __init__(self) -> None:
        super().__init__()
        self.cmake_options: T.List[str] = []
        self.target_options = TargetOptions()
        self.methods.update({'add_cmake_defines': self.add_cmake_defines, 'set_override_option': self.set_override_option, 'set_install': self.set_install, 'append_compile_args': self.append_compile_args, 'append_link_args': self.append_link_args, 'clear': self.clear})

    def _get_opts(self, kwargs: dict) -> SingleTargetOptions:
        if 'target' in kwargs:
            return self.target_options[kwargs['target']]
        return self.target_options.global_options

    @typed_pos_args('subproject_options.add_cmake_defines', varargs=dict)
    @noKwargs
    def add_cmake_defines(self, state: ModuleState, args: T.Tuple[T.List[T.Dict[str, TYPE_var]]], kwargs: TYPE_kwargs) -> None:
        self.cmake_options += cmake_defines_to_args(args[0])

    @typed_pos_args('subproject_options.set_override_option', str, str)
    @permittedKwargs({'target'})
    def set_override_option(self, state: ModuleState, args: T.Tuple[str, str], kwargs: TYPE_kwargs) -> None:
        self._get_opts(kwargs).set_opt(args[0], args[1])

    @typed_pos_args('subproject_options.set_install', bool)
    @permittedKwargs({'target'})
    def set_install(self, state: ModuleState, args: T.Tuple[bool], kwargs: TYPE_kwargs) -> None:
        self._get_opts(kwargs).set_install(args[0])

    @typed_pos_args('subproject_options.append_compile_args', str, varargs=str, min_varargs=1)
    @permittedKwargs({'target'})
    def append_compile_args(self, state: ModuleState, args: T.Tuple[str, T.List[str]], kwargs: TYPE_kwargs) -> None:
        self._get_opts(kwargs).append_args(args[0], args[1])

    @typed_pos_args('subproject_options.append_compile_args', varargs=str, min_varargs=1)
    @permittedKwargs({'target'})
    def append_link_args(self, state: ModuleState, args: T.Tuple[T.List[str]], kwargs: TYPE_kwargs) -> None:
        self._get_opts(kwargs).append_link_args(args[0])

    @noPosargs
    @noKwargs
    def clear(self, state: ModuleState, args: TYPE_var, kwargs: TYPE_kwargs) -> None:
        self.cmake_options.clear()
        self.target_options = TargetOptions()