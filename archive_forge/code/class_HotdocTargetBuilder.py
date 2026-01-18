from __future__ import annotations
import os, subprocess
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build, mesonlib, mlog
from ..build import CustomTarget, CustomTargetIndex
from ..dependencies import Dependency, InternalDependency
from ..interpreterbase import (
from ..interpreter.interpreterobjects import _CustomTargetHolder
from ..interpreter.type_checking import NoneType
from ..mesonlib import File, MesonException
from ..programs import ExternalProgram
class HotdocTargetBuilder:

    def __init__(self, name: str, state: ModuleState, hotdoc: HotdocExternalProgram, interpreter: Interpreter, kwargs):
        self.hotdoc = hotdoc
        self.build_by_default = kwargs.pop('build_by_default', False)
        self.kwargs = kwargs
        self.name = name
        self.state = state
        self.interpreter = interpreter
        self.include_paths: mesonlib.OrderedSet[str] = mesonlib.OrderedSet()
        self.builddir = state.environment.get_build_dir()
        self.sourcedir = state.environment.get_source_dir()
        self.subdir = state.subdir
        self.build_command = state.environment.get_build_command()
        self.cmd: T.List[TYPE_var] = ['conf', '--project-name', name, '--disable-incremental-build', '--output', os.path.join(self.builddir, self.subdir, self.name + '-doc')]
        self._extra_extension_paths = set()
        self.extra_assets = set()
        self.extra_depends = []
        self._subprojects = []

    def process_known_arg(self, option: str, argname: T.Optional[str]=None, value_processor: T.Optional[T.Callable]=None) -> None:
        if not argname:
            argname = option.strip('-').replace('-', '_')
        value = self.kwargs.pop(argname)
        if value is not None and value_processor:
            value = value_processor(value)
        self.set_arg_value(option, value)

    def set_arg_value(self, option: str, value: TYPE_var) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            if value:
                self.cmd.append(option)
        elif isinstance(value, list):
            if value:
                escaped_value = []
                for e in value:
                    if isinstance(e, str) and e.startswith('-'):
                        escaped_value += [' %s' % e]
                    else:
                        escaped_value += [e]
                if option:
                    self.cmd.extend([option] + escaped_value)
                else:
                    self.cmd.extend(escaped_value)
        elif isinstance(value, str):
            self.cmd.extend([f'{option}={value}'])
        else:
            self.cmd.extend([option, value])

    def check_extra_arg_type(self, arg: str, value: TYPE_var) -> None:
        if isinstance(value, list):
            for v in value:
                self.check_extra_arg_type(arg, v)
            return
        valid_types = (str, bool, File, build.IncludeDirs, CustomTarget, CustomTargetIndex, build.BuildTarget)
        if not isinstance(value, valid_types):
            raise InvalidArguments('Argument "{}={}" should be of type: {}.'.format(arg, value, [t.__name__ for t in valid_types]))

    def process_extra_args(self) -> None:
        for arg, value in self.kwargs.items():
            option = '--' + arg.replace('_', '-')
            self.check_extra_arg_type(arg, value)
            self.set_arg_value(option, value)

    def get_value(self, types, argname, default=None, value_processor=None, mandatory=False, force_list=False):
        if not isinstance(types, list):
            types = [types]
        try:
            uvalue = value = self.kwargs.pop(argname)
            if value_processor:
                value = value_processor(value)
            for t in types:
                if isinstance(value, t):
                    if force_list and (not isinstance(value, list)):
                        return ([value], uvalue)
                    return (value, uvalue)
            raise MesonException(f'{argname} field value {value} is not valid, valid types are {types}')
        except KeyError:
            if mandatory:
                raise MesonException(f'{argname} mandatory field not found')
            if default is not None:
                return (default, default)
        return (None, None)

    def add_extension_paths(self, paths: T.Union[T.List[str], T.Set[str]]) -> None:
        for path in paths:
            if path in self._extra_extension_paths:
                continue
            self._extra_extension_paths.add(path)
            self.cmd.extend(['--extra-extension-path', path])

    def replace_dirs_in_string(self, string: str) -> str:
        return string.replace('@SOURCE_ROOT@', self.sourcedir).replace('@BUILD_ROOT@', self.builddir)

    def process_gi_c_source_roots(self) -> None:
        if self.hotdoc.run_hotdoc(['--has-extension=gi-extension']) != 0:
            return
        value = self.kwargs.pop('gi_c_source_roots')
        value.extend([os.path.join(self.sourcedir, self.state.root_subdir), os.path.join(self.builddir, self.state.root_subdir)])
        self.cmd += ['--gi-c-source-roots'] + value

    def process_dependencies(self, deps: T.List[T.Union[Dependency, build.StaticLibrary, build.SharedLibrary, CustomTarget, CustomTargetIndex]]) -> T.List[str]:
        cflags = set()
        for dep in mesonlib.listify(ensure_list(deps)):
            if isinstance(dep, InternalDependency):
                inc_args = self.state.get_include_args(dep.include_directories)
                cflags.update([self.replace_dirs_in_string(x) for x in inc_args])
                cflags.update(self.process_dependencies(dep.libraries))
                cflags.update(self.process_dependencies(dep.sources))
                cflags.update(self.process_dependencies(dep.ext_deps))
            elif isinstance(dep, Dependency):
                cflags.update(dep.get_compile_args())
            elif isinstance(dep, (build.StaticLibrary, build.SharedLibrary)):
                self.extra_depends.append(dep)
                for incd in dep.get_include_dirs():
                    cflags.update(incd.get_incdirs())
            elif isinstance(dep, HotdocTarget):
                self.process_dependencies(dep.get_target_dependencies())
                self._subprojects.extend(dep.subprojects)
                self.process_dependencies(dep.subprojects)
                self.include_paths.add(os.path.join(self.builddir, dep.hotdoc_conf.subdir))
                self.cmd += ['--extra-assets=' + p for p in dep.extra_assets]
                self.add_extension_paths(dep.extra_extension_paths)
            elif isinstance(dep, (CustomTarget, build.BuildTarget)):
                self.extra_depends.append(dep)
            elif isinstance(dep, CustomTargetIndex):
                self.extra_depends.append(dep.target)
        return [f.strip('-I') for f in cflags]

    def process_extra_assets(self) -> None:
        self._extra_assets = self.kwargs.pop('extra_assets')
        for assets_path in self._extra_assets:
            self.cmd.extend(['--extra-assets', assets_path])

    def process_subprojects(self) -> None:
        value = self.kwargs.pop('subprojects')
        self.process_dependencies(value)
        self._subprojects.extend(value)

    def flatten_config_command(self) -> T.List[str]:
        cmd = []
        for arg in mesonlib.listify(self.cmd, flatten=True):
            if isinstance(arg, File):
                arg = arg.absolute_path(self.state.environment.get_source_dir(), self.state.environment.get_build_dir())
            elif isinstance(arg, build.IncludeDirs):
                for inc_dir in arg.get_incdirs():
                    cmd.append(os.path.join(self.sourcedir, arg.get_curdir(), inc_dir))
                    cmd.append(os.path.join(self.builddir, arg.get_curdir(), inc_dir))
                continue
            elif isinstance(arg, (build.BuildTarget, CustomTarget)):
                self.extra_depends.append(arg)
                arg = self.interpreter.backend.get_target_filename_abs(arg)
            elif isinstance(arg, CustomTargetIndex):
                self.extra_depends.append(arg.target)
                arg = self.interpreter.backend.get_target_filename_abs(arg)
            cmd.append(arg)
        return cmd

    def generate_hotdoc_config(self) -> None:
        cwd = os.path.abspath(os.curdir)
        ncwd = os.path.join(self.sourcedir, self.subdir)
        mlog.log('Generating Hotdoc configuration for: ', mlog.bold(self.name))
        os.chdir(ncwd)
        if self.hotdoc.run_hotdoc(self.flatten_config_command()) != 0:
            raise MesonException('hotdoc failed to configure')
        os.chdir(cwd)

    def ensure_file(self, value: T.Union[str, File, CustomTarget, CustomTargetIndex]) -> T.Union[File, CustomTarget, CustomTargetIndex]:
        if isinstance(value, list):
            res = []
            for val in value:
                res.append(self.ensure_file(val))
            return res
        if isinstance(value, str):
            return File.from_source_file(self.sourcedir, self.subdir, value)
        return value

    def ensure_dir(self, value: str) -> str:
        if os.path.isabs(value):
            _dir = value
        else:
            _dir = os.path.join(self.sourcedir, self.subdir, value)
        if not os.path.isdir(_dir):
            raise InvalidArguments(f'"{_dir}" is not a directory.')
        return os.path.relpath(_dir, os.path.join(self.builddir, self.subdir))

    def check_forbidden_args(self) -> None:
        for arg in ['conf_file']:
            if arg in self.kwargs:
                raise InvalidArguments(f'Argument "{arg}" is forbidden.')

    def make_targets(self) -> T.Tuple[HotdocTarget, mesonlib.ExecutableSerialisation]:
        self.check_forbidden_args()
        self.process_known_arg('--index', value_processor=self.ensure_file)
        self.process_known_arg('--project-version')
        self.process_known_arg('--sitemap', value_processor=self.ensure_file)
        self.process_known_arg('--html-extra-theme', value_processor=self.ensure_dir)
        self.include_paths.update((self.ensure_dir(v) for v in self.kwargs.pop('include_paths')))
        self.process_known_arg('--c-include-directories', argname='dependencies', value_processor=self.process_dependencies)
        self.process_gi_c_source_roots()
        self.process_extra_assets()
        self.add_extension_paths(self.kwargs.pop('extra_extension_paths'))
        self.process_subprojects()
        self.extra_depends.extend(self.kwargs.pop('depends'))
        install = self.kwargs.pop('install')
        self.process_extra_args()
        fullname = self.name + '-doc'
        hotdoc_config_name = fullname + '.json'
        hotdoc_config_path = os.path.join(self.builddir, self.subdir, hotdoc_config_name)
        with open(hotdoc_config_path, 'w', encoding='utf-8') as f:
            f.write('{}')
        self.cmd += ['--conf-file', hotdoc_config_path]
        self.include_paths.add(os.path.join(self.builddir, self.subdir))
        self.include_paths.add(os.path.join(self.sourcedir, self.subdir))
        depfile = os.path.join(self.builddir, self.subdir, self.name + '.deps')
        self.cmd += ['--deps-file-dest', depfile]
        for path in self.include_paths:
            self.cmd.extend(['--include-path', path])
        if self.state.environment.coredata.get_option(mesonlib.OptionKey('werror', subproject=self.state.subproject)):
            self.cmd.append('--fatal-warnings')
        self.generate_hotdoc_config()
        target_cmd = self.build_command + ['--internal', 'hotdoc'] + self.hotdoc.get_command() + ['run', '--conf-file', hotdoc_config_name] + ['--builddir', os.path.join(self.builddir, self.subdir)]
        target = HotdocTarget(fullname, subdir=self.subdir, subproject=self.state.subproject, environment=self.state.environment, hotdoc_conf=File.from_built_file(self.subdir, hotdoc_config_name), extra_extension_paths=self._extra_extension_paths, extra_assets=self._extra_assets, subprojects=self._subprojects, command=target_cmd, extra_depends=self.extra_depends, outputs=[fullname], sources=[], depfile=os.path.basename(depfile), build_by_default=self.build_by_default)
        install_script = None
        if install:
            datadir = os.path.join(self.state.get_option('prefix'), self.state.get_option('datadir'))
            devhelp = self.kwargs.get('devhelp_activate', False)
            if not isinstance(devhelp, bool):
                FeatureDeprecated.single_use('hotdoc.generate_doc() devhelp_activate must be boolean', '1.1.0', self.state.subproject)
                devhelp = False
            if devhelp:
                install_from = os.path.join(fullname, 'devhelp')
                install_to = os.path.join(datadir, 'devhelp')
            else:
                install_from = os.path.join(fullname, 'html')
                install_to = os.path.join(datadir, 'doc', self.name, 'html')
            install_script = self.state.backend.get_executable_serialisation(self.build_command + ['--internal', 'hotdoc', '--install', install_from, '--docdir', install_to, '--name', self.name, '--builddir', os.path.join(self.builddir, self.subdir)] + self.hotdoc.get_command() + ['run', '--conf-file', hotdoc_config_name])
            install_script.tag = 'doc'
        return (target, install_script)