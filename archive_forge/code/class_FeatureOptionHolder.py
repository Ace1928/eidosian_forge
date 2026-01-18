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
class FeatureOptionHolder(ObjectHolder[coredata.UserFeatureOption]):

    def __init__(self, option: coredata.UserFeatureOption, interpreter: 'Interpreter'):
        super().__init__(option, interpreter)
        if option and option.is_auto():
            auto = T.cast('coredata.UserFeatureOption', self.env.coredata.options[OptionKey('auto_features')])
            self.held_object = copy.copy(auto)
            self.held_object.name = option.name
        self.methods.update({'enabled': self.enabled_method, 'disabled': self.disabled_method, 'allowed': self.allowed_method, 'auto': self.auto_method, 'require': self.require_method, 'disable_auto_if': self.disable_auto_if_method, 'enable_auto_if': self.enable_auto_if_method, 'disable_if': self.disable_if_method, 'enable_if': self.enable_if_method})

    @property
    def value(self) -> str:
        return 'disabled' if not self.held_object else self.held_object.value

    def as_disabled(self) -> coredata.UserFeatureOption:
        disabled = copy.deepcopy(self.held_object)
        disabled.value = 'disabled'
        return disabled

    def as_enabled(self) -> coredata.UserFeatureOption:
        enabled = copy.deepcopy(self.held_object)
        enabled.value = 'enabled'
        return enabled

    @noPosargs
    @noKwargs
    def enabled_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> bool:
        return self.value == 'enabled'

    @noPosargs
    @noKwargs
    def disabled_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> bool:
        return self.value == 'disabled'

    @noPosargs
    @noKwargs
    @FeatureNew('feature_option.allowed()', '0.59.0')
    def allowed_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> bool:
        return self.value != 'disabled'

    @noPosargs
    @noKwargs
    def auto_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> bool:
        return self.value == 'auto'

    def _disable_if(self, condition: bool, message: T.Optional[str]) -> coredata.UserFeatureOption:
        if not condition:
            return copy.deepcopy(self.held_object)
        if self.value == 'enabled':
            err_msg = f'Feature {self.held_object.name} cannot be enabled'
            if message:
                err_msg += f': {message}'
            raise InterpreterException(err_msg)
        return self.as_disabled()

    @FeatureNew('feature_option.require()', '0.59.0')
    @typed_pos_args('feature_option.require', bool)
    @typed_kwargs('feature_option.require', _ERROR_MSG_KW)
    def require_method(self, args: T.Tuple[bool], kwargs: 'kwargs.FeatureOptionRequire') -> coredata.UserFeatureOption:
        return self._disable_if(not args[0], kwargs['error_message'])

    @FeatureNew('feature_option.disable_if()', '1.1.0')
    @typed_pos_args('feature_option.disable_if', bool)
    @typed_kwargs('feature_option.disable_if', _ERROR_MSG_KW)
    def disable_if_method(self, args: T.Tuple[bool], kwargs: 'kwargs.FeatureOptionRequire') -> coredata.UserFeatureOption:
        return self._disable_if(args[0], kwargs['error_message'])

    @FeatureNew('feature_option.enable_if()', '1.1.0')
    @typed_pos_args('feature_option.enable_if', bool)
    @typed_kwargs('feature_option.enable_if', _ERROR_MSG_KW)
    def enable_if_method(self, args: T.Tuple[bool], kwargs: 'kwargs.FeatureOptionRequire') -> coredata.UserFeatureOption:
        if not args[0]:
            return copy.deepcopy(self.held_object)
        if self.value == 'disabled':
            err_msg = f'Feature {self.held_object.name} cannot be disabled'
            if kwargs['error_message']:
                err_msg += f': {kwargs['error_message']}'
            raise InterpreterException(err_msg)
        return self.as_enabled()

    @FeatureNew('feature_option.disable_auto_if()', '0.59.0')
    @noKwargs
    @typed_pos_args('feature_option.disable_auto_if', bool)
    def disable_auto_if_method(self, args: T.Tuple[bool], kwargs: TYPE_kwargs) -> coredata.UserFeatureOption:
        return copy.deepcopy(self.held_object) if self.value != 'auto' or not args[0] else self.as_disabled()

    @FeatureNew('feature_option.enable_auto_if()', '1.1.0')
    @noKwargs
    @typed_pos_args('feature_option.enable_auto_if', bool)
    def enable_auto_if_method(self, args: T.Tuple[bool], kwargs: TYPE_kwargs) -> coredata.UserFeatureOption:
        return self.as_enabled() if self.value == 'auto' and args[0] else copy.deepcopy(self.held_object)