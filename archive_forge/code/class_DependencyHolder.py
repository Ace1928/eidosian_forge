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
class DependencyHolder(ObjectHolder[Dependency]):

    def __init__(self, dep: Dependency, interpreter: 'Interpreter'):
        super().__init__(dep, interpreter)
        self.methods.update({'found': self.found_method, 'type_name': self.type_name_method, 'version': self.version_method, 'name': self.name_method, 'get_pkgconfig_variable': self.pkgconfig_method, 'get_configtool_variable': self.configtool_method, 'get_variable': self.variable_method, 'partial_dependency': self.partial_dependency_method, 'include_type': self.include_type_method, 'as_system': self.as_system_method, 'as_link_whole': self.as_link_whole_method})

    def found(self) -> bool:
        return self.found_method([], {})

    @noPosargs
    @noKwargs
    def type_name_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self.held_object.type_name

    @noPosargs
    @noKwargs
    def found_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> bool:
        if self.held_object.type_name == 'internal':
            return True
        return self.held_object.found()

    @noPosargs
    @noKwargs
    def version_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self.held_object.get_version()

    @noPosargs
    @noKwargs
    def name_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self.held_object.get_name()

    @FeatureDeprecated('dependency.get_pkgconfig_variable', '0.56.0', 'use dependency.get_variable(pkgconfig : ...) instead')
    @typed_pos_args('dependency.get_pkgconfig_variable', str)
    @typed_kwargs('dependency.get_pkgconfig_variable', KwargInfo('default', str, default=''), PKGCONFIG_DEFINE_KW.evolve(name='define_variable'))
    def pkgconfig_method(self, args: T.Tuple[str], kwargs: 'kwargs.DependencyPkgConfigVar') -> str:
        from ..dependencies.pkgconfig import PkgConfigDependency
        if not isinstance(self.held_object, PkgConfigDependency):
            raise InvalidArguments(f'{self.held_object.get_name()!r} is not a pkgconfig dependency')
        if kwargs['define_variable'] and len(kwargs['define_variable']) > 1:
            FeatureNew.single_use('dependency.get_pkgconfig_variable keyword argument "define_variable"  with more than one pair', '1.3.0', self.subproject, location=self.current_node)
        return self.held_object.get_variable(pkgconfig=args[0], default_value=kwargs['default'], pkgconfig_define=kwargs['define_variable'])

    @FeatureNew('dependency.get_configtool_variable', '0.44.0')
    @FeatureDeprecated('dependency.get_configtool_variable', '0.56.0', 'use dependency.get_variable(configtool : ...) instead')
    @noKwargs
    @typed_pos_args('dependency.get_config_tool_variable', str)
    def configtool_method(self, args: T.Tuple[str], kwargs: TYPE_kwargs) -> str:
        from ..dependencies.configtool import ConfigToolDependency
        if not isinstance(self.held_object, ConfigToolDependency):
            raise InvalidArguments(f'{self.held_object.get_name()!r} is not a config-tool dependency')
        return self.held_object.get_variable(configtool=args[0], default_value='')

    @FeatureNew('dependency.partial_dependency', '0.46.0')
    @noPosargs
    @typed_kwargs('dependency.partial_dependency', *_PARTIAL_DEP_KWARGS)
    def partial_dependency_method(self, args: T.List[TYPE_nvar], kwargs: 'kwargs.DependencyMethodPartialDependency') -> Dependency:
        pdep = self.held_object.get_partial_dependency(**kwargs)
        return pdep

    @FeatureNew('dependency.get_variable', '0.51.0')
    @typed_pos_args('dependency.get_variable', optargs=[str])
    @typed_kwargs('dependency.get_variable', KwargInfo('cmake', (str, NoneType)), KwargInfo('pkgconfig', (str, NoneType)), KwargInfo('configtool', (str, NoneType)), KwargInfo('internal', (str, NoneType), since='0.54.0'), KwargInfo('default_value', (str, NoneType)), PKGCONFIG_DEFINE_KW)
    def variable_method(self, args: T.Tuple[T.Optional[str]], kwargs: 'kwargs.DependencyGetVariable') -> str:
        default_varname = args[0]
        if default_varname is not None:
            FeatureNew('Positional argument to dependency.get_variable()', '0.58.0').use(self.subproject, self.current_node)
        if kwargs['pkgconfig_define'] and len(kwargs['pkgconfig_define']) > 1:
            FeatureNew.single_use('dependency.get_variable keyword argument "pkgconfig_define" with more than one pair', '1.3.0', self.subproject, 'In previous versions, this silently returned a malformed value.', self.current_node)
        return self.held_object.get_variable(cmake=kwargs['cmake'] or default_varname, pkgconfig=kwargs['pkgconfig'] or default_varname, configtool=kwargs['configtool'] or default_varname, internal=kwargs['internal'] or default_varname, default_value=kwargs['default_value'], pkgconfig_define=kwargs['pkgconfig_define'])

    @FeatureNew('dependency.include_type', '0.52.0')
    @noPosargs
    @noKwargs
    def include_type_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
        return self.held_object.get_include_type()

    @FeatureNew('dependency.as_system', '0.52.0')
    @noKwargs
    @typed_pos_args('dependency.as_system', optargs=[str])
    def as_system_method(self, args: T.Tuple[T.Optional[str]], kwargs: TYPE_kwargs) -> Dependency:
        return self.held_object.generate_system_dependency(args[0] or 'system')

    @FeatureNew('dependency.as_link_whole', '0.56.0')
    @noKwargs
    @noPosargs
    def as_link_whole_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> Dependency:
        if not isinstance(self.held_object, InternalDependency):
            raise InterpreterException('as_link_whole method is only supported on declare_dependency() objects')
        new_dep = self.held_object.generate_link_whole_dependency()
        return new_dep