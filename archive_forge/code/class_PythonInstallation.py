from __future__ import annotations
import copy, json, os, shutil, re
import typing as T
from . import ExtensionModule, ModuleInfo
from .. import mesonlib
from .. import mlog
from ..coredata import UserFeatureOption
from ..build import known_shmod_kwargs, CustomTarget, CustomTargetIndex, BuildTarget, GeneratedList, StructuredSources, ExtractedObjects, SharedModule
from ..dependencies import NotFoundDependency
from ..dependencies.detect import get_dep_identifier, find_external_dependency
from ..dependencies.python import BasicPythonExternalProgram, python_factory, _PythonDependencyBase
from ..interpreter import extract_required_kwarg, permitted_dependency_kwargs, primitives as P_OBJ
from ..interpreter.interpreterobjects import _ExternalProgramHolder
from ..interpreter.type_checking import NoneType, PRESERVE_PATH_KW, SHARED_MOD_KWS
from ..interpreterbase import (
from ..mesonlib import MachineChoice, OptionKey
from ..programs import ExternalProgram, NonExistingExternalProgram
class PythonInstallation(_ExternalProgramHolder['PythonExternalProgram']):

    def __init__(self, python: 'PythonExternalProgram', interpreter: 'Interpreter'):
        _ExternalProgramHolder.__init__(self, python, interpreter)
        info = python.info
        prefix = self.interpreter.environment.coredata.get_option(mesonlib.OptionKey('prefix'))
        assert isinstance(prefix, str), 'for mypy'
        self.variables = info['variables']
        self.suffix = info['suffix']
        self.limited_api_suffix = info['limited_api_suffix']
        self.paths = info['paths']
        self.pure = python.pure
        self.platlib_install_path = os.path.join(prefix, python.platlib)
        self.purelib_install_path = os.path.join(prefix, python.purelib)
        self.version = info['version']
        self.platform = info['platform']
        self.is_pypy = info['is_pypy']
        self.link_libpython = info['link_libpython']
        self.methods.update({'extension_module': self.extension_module_method, 'dependency': self.dependency_method, 'install_sources': self.install_sources_method, 'get_install_dir': self.get_install_dir_method, 'language_version': self.language_version_method, 'found': self.found_method, 'has_path': self.has_path_method, 'get_path': self.get_path_method, 'has_variable': self.has_variable_method, 'get_variable': self.get_variable_method, 'path': self.path_method})

    @permittedKwargs(mod_kwargs)
    @typed_pos_args('python.extension_module', str, varargs=(str, mesonlib.File, CustomTarget, CustomTargetIndex, GeneratedList, StructuredSources, ExtractedObjects, BuildTarget))
    @typed_kwargs('python.extension_module', *_MOD_KWARGS, _DEFAULTABLE_SUBDIR_KW, _LIMITED_API_KW, allow_unknown=True)
    def extension_module_method(self, args: T.Tuple[str, T.List[BuildTargetSource]], kwargs: ExtensionModuleKw) -> 'SharedModule':
        if 'install_dir' in kwargs:
            if kwargs['subdir'] is not None:
                raise InvalidArguments('"subdir" and "install_dir" are mutually exclusive')
        else:
            subdir = kwargs.pop('subdir') or ''
            kwargs['install_dir'] = self._get_install_dir_impl(False, subdir)
        target_suffix = self.suffix
        new_deps = mesonlib.extract_as_list(kwargs, 'dependencies')
        pydep = next((dep for dep in new_deps if isinstance(dep, _PythonDependencyBase)), None)
        if pydep is None:
            pydep = self._dependency_method_impl({})
            if not pydep.found():
                raise mesonlib.MesonException('Python dependency not found')
            new_deps.append(pydep)
            FeatureNew.single_use('python_installation.extension_module with implicit dependency on python', '0.63.0', self.subproject, 'use python_installation.dependency()', self.current_node)
        limited_api_version = kwargs.pop('limited_api')
        allow_limited_api = self.interpreter.environment.coredata.get_option(OptionKey('allow_limited_api', module='python'))
        if limited_api_version != '' and allow_limited_api:
            target_suffix = self.limited_api_suffix
            limited_api_version_hex = self._convert_api_version_to_py_version_hex(limited_api_version, pydep.version)
            limited_api_definition = f'-DPy_LIMITED_API={limited_api_version_hex}'
            new_c_args = mesonlib.extract_as_list(kwargs, 'c_args')
            new_c_args.append(limited_api_definition)
            kwargs['c_args'] = new_c_args
            new_cpp_args = mesonlib.extract_as_list(kwargs, 'cpp_args')
            new_cpp_args.append(limited_api_definition)
            kwargs['cpp_args'] = new_cpp_args
            for_machine = kwargs['native']
            compilers = self.interpreter.environment.coredata.compilers[for_machine]
            if any((compiler.get_id() == 'msvc' for compiler in compilers.values())):
                pydep_copy = copy.copy(pydep)
                pydep_copy.find_libpy_windows(self.env, limited_api=True)
                if not pydep_copy.found():
                    raise mesonlib.MesonException('Python dependency supporting limited API not found')
                new_deps.remove(pydep)
                new_deps.append(pydep_copy)
                pyver = pydep.version.replace('.', '')
                python_windows_debug_link_exception = f'/NODEFAULTLIB:python{pyver}_d.lib'
                python_windows_release_link_exception = f'/NODEFAULTLIB:python{pyver}.lib'
                new_link_args = mesonlib.extract_as_list(kwargs, 'link_args')
                is_debug = self.interpreter.environment.coredata.options[OptionKey('debug')].value
                if is_debug:
                    new_link_args.append(python_windows_debug_link_exception)
                else:
                    new_link_args.append(python_windows_release_link_exception)
                kwargs['link_args'] = new_link_args
        kwargs['dependencies'] = new_deps
        split, target_suffix = target_suffix.rsplit('.', 1)
        args = (args[0] + split, args[1])
        kwargs['name_prefix'] = ''
        kwargs['name_suffix'] = target_suffix
        if kwargs['gnu_symbol_visibility'] == '' and (self.is_pypy or mesonlib.version_compare(self.version, '>=3.9')):
            kwargs['gnu_symbol_visibility'] = 'inlineshidden'
        return self.interpreter.build_target(self.current_node, args, kwargs, SharedModule)

    def _convert_api_version_to_py_version_hex(self, api_version: str, detected_version: str) -> str:
        python_api_version_format = re.compile('[0-9]\\.[0-9]{1,2}')
        decimal_match = python_api_version_format.fullmatch(api_version)
        if not decimal_match:
            raise InvalidArguments(f'Python API version invalid: "{api_version}".')
        if mesonlib.version_compare(api_version, '<3.2'):
            raise InvalidArguments(f'Python Limited API version invalid: {api_version} (must be greater than 3.2)')
        if mesonlib.version_compare(api_version, '>' + detected_version):
            raise InvalidArguments(f'Python Limited API version too high: {api_version} (detected {detected_version})')
        version_components = api_version.split('.')
        major = int(version_components[0])
        minor = int(version_components[1])
        return '0x{:02x}{:02x}0000'.format(major, minor)

    def _dependency_method_impl(self, kwargs: TYPE_kwargs) -> Dependency:
        for_machine = self.interpreter.machine_from_native_kwarg(kwargs)
        identifier = get_dep_identifier(self._full_path(), kwargs)
        dep = self.interpreter.coredata.deps[for_machine].get(identifier)
        if dep is not None:
            return dep
        new_kwargs = kwargs.copy()
        new_kwargs['required'] = False
        candidates = python_factory(self.interpreter.environment, for_machine, new_kwargs, self.held_object)
        dep = find_external_dependency('python', self.interpreter.environment, new_kwargs, candidates)
        self.interpreter.coredata.deps[for_machine].put(identifier, dep)
        return dep

    @disablerIfNotFound
    @permittedKwargs(permitted_dependency_kwargs | {'embed'})
    @FeatureNewKwargs('python_installation.dependency', '0.53.0', ['embed'])
    @noPosargs
    def dependency_method(self, args: T.List['TYPE_var'], kwargs: 'TYPE_kwargs') -> 'Dependency':
        disabled, required, feature = extract_required_kwarg(kwargs, self.subproject)
        if disabled:
            mlog.log('Dependency', mlog.bold('python'), 'skipped: feature', mlog.bold(feature), 'disabled')
            return NotFoundDependency('python', self.interpreter.environment)
        else:
            dep = self._dependency_method_impl(kwargs)
            if required and (not dep.found()):
                raise mesonlib.MesonException('Python dependency not found')
            return dep

    @typed_pos_args('install_data', varargs=(str, mesonlib.File))
    @typed_kwargs('python_installation.install_sources', _PURE_KW, _SUBDIR_KW, PRESERVE_PATH_KW, KwargInfo('install_tag', (str, NoneType), since='0.60.0'))
    def install_sources_method(self, args: T.Tuple[T.List[T.Union[str, mesonlib.File]]], kwargs: 'PyInstallKw') -> 'Data':
        self.held_object.run_bytecompile[self.version] = True
        tag = kwargs['install_tag'] or 'python-runtime'
        pure = kwargs['pure'] if kwargs['pure'] is not None else self.pure
        install_dir = self._get_install_dir_impl(pure, kwargs['subdir'])
        return self.interpreter.install_data_impl(self.interpreter.source_strings_to_files(args[0]), install_dir, mesonlib.FileMode(), rename=None, tag=tag, install_data_type='python', preserve_path=kwargs['preserve_path'])

    @noPosargs
    @typed_kwargs('python_installation.install_dir', _PURE_KW, _SUBDIR_KW)
    def get_install_dir_method(self, args: T.List['TYPE_var'], kwargs: 'PyInstallKw') -> str:
        self.held_object.run_bytecompile[self.version] = True
        pure = kwargs['pure'] if kwargs['pure'] is not None else self.pure
        return self._get_install_dir_impl(pure, kwargs['subdir'])

    def _get_install_dir_impl(self, pure: bool, subdir: str) -> P_OBJ.OptionString:
        if pure:
            base = self.purelib_install_path
            name = '{py_purelib}'
        else:
            base = self.platlib_install_path
            name = '{py_platlib}'
        return P_OBJ.OptionString(os.path.join(base, subdir), os.path.join(name, subdir))

    @noPosargs
    @noKwargs
    def language_version_method(self, args: T.List['TYPE_var'], kwargs: 'TYPE_kwargs') -> str:
        return self.version

    @typed_pos_args('python_installation.has_path', str)
    @noKwargs
    def has_path_method(self, args: T.Tuple[str], kwargs: 'TYPE_kwargs') -> bool:
        return args[0] in self.paths

    @typed_pos_args('python_installation.get_path', str, optargs=[object])
    @noKwargs
    def get_path_method(self, args: T.Tuple[str, T.Optional['TYPE_var']], kwargs: 'TYPE_kwargs') -> 'TYPE_var':
        path_name, fallback = args
        try:
            return self.paths[path_name]
        except KeyError:
            if fallback is not None:
                return fallback
            raise InvalidArguments(f'{path_name} is not a valid path name')

    @typed_pos_args('python_installation.has_variable', str)
    @noKwargs
    def has_variable_method(self, args: T.Tuple[str], kwargs: 'TYPE_kwargs') -> bool:
        return args[0] in self.variables

    @typed_pos_args('python_installation.get_variable', str, optargs=[object])
    @noKwargs
    def get_variable_method(self, args: T.Tuple[str, T.Optional['TYPE_var']], kwargs: 'TYPE_kwargs') -> 'TYPE_var':
        var_name, fallback = args
        try:
            return self.variables[var_name]
        except KeyError:
            if fallback is not None:
                return fallback
            raise InvalidArguments(f'{var_name} is not a valid variable name')

    @noPosargs
    @noKwargs
    @FeatureNew('Python module path method', '0.50.0')
    def path_method(self, args: T.List['TYPE_var'], kwargs: 'TYPE_kwargs') -> str:
        return super().path_method(args, kwargs)