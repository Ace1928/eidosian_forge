import logging
import io
import os
import shutil
import sys
import traceback
from contextlib import suppress
from enum import Enum
from inspect import cleandoc
from itertools import chain, starmap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
from .. import (
from ..discovery import find_package_path
from ..dist import Distribution
from ..warnings import (
from .build_py import build_py as build_py_cls
import sys
from importlib.machinery import ModuleSpec, PathFinder
from importlib.machinery import all_suffixes as module_suffixes
from importlib.util import spec_from_file_location
from itertools import chain
from pathlib import Path
class editable_wheel(Command):
    """Build 'editable' wheel for development.
    This command is private and reserved for internal use of setuptools,
    users should rely on ``setuptools.build_meta`` APIs.
    """
    description = 'DO NOT CALL DIRECTLY, INTERNAL ONLY: create PEP 660 editable wheel'
    user_options = [('dist-dir=', 'd', 'directory to put final built distributions in'), ('dist-info-dir=', 'I', 'path to a pre-build .dist-info directory'), ('mode=', None, cleandoc(_EditableMode.__doc__ or ''))]

    def initialize_options(self):
        self.dist_dir = None
        self.dist_info_dir = None
        self.project_dir = None
        self.mode = None

    def finalize_options(self):
        dist = self.distribution
        self.project_dir = dist.src_root or os.curdir
        self.package_dir = dist.package_dir or {}
        self.dist_dir = Path(self.dist_dir or os.path.join(self.project_dir, 'dist'))

    def run(self):
        try:
            self.dist_dir.mkdir(exist_ok=True)
            self._ensure_dist_info()
            self.reinitialize_command('bdist_wheel')
            bdist_wheel = self.get_finalized_command('bdist_wheel')
            bdist_wheel.write_wheelfile(self.dist_info_dir)
            self._create_wheel_file(bdist_wheel)
        except Exception:
            traceback.print_exc()
            project = self.distribution.name or self.distribution.get_name()
            _DebuggingTips.emit(project=project)
            raise

    def _ensure_dist_info(self):
        if self.dist_info_dir is None:
            dist_info = self.reinitialize_command('dist_info')
            dist_info.output_dir = self.dist_dir
            dist_info.ensure_finalized()
            dist_info.run()
            self.dist_info_dir = dist_info.dist_info_dir
        else:
            assert str(self.dist_info_dir).endswith('.dist-info')
            assert Path(self.dist_info_dir, 'METADATA').exists()

    def _install_namespaces(self, installation_dir, pth_prefix):
        dist = self.distribution
        if not dist.namespace_packages:
            return
        src_root = Path(self.project_dir, self.package_dir.get('', '.')).resolve()
        installer = _NamespaceInstaller(dist, installation_dir, pth_prefix, src_root)
        installer.install_namespaces()

    def _find_egg_info_dir(self) -> Optional[str]:
        parent_dir = Path(self.dist_info_dir).parent if self.dist_info_dir else Path()
        candidates = map(str, parent_dir.glob('*.egg-info'))
        return next(candidates, None)

    def _configure_build(self, name: str, unpacked_wheel: _Path, build_lib: _Path, tmp_dir: _Path):
        """Configure commands to behave in the following ways:

        - Build commands can write to ``build_lib`` if they really want to...
          (but this folder is expected to be ignored and modules are expected to live
          in the project directory...)
        - Binary extensions should be built in-place (editable_mode = True)
        - Data/header/script files are not part of the "editable" specification
          so they are written directly to the unpacked_wheel directory.
        """
        dist = self.distribution
        wheel = str(unpacked_wheel)
        build_lib = str(build_lib)
        data = str(Path(unpacked_wheel, f'{name}.data', 'data'))
        headers = str(Path(unpacked_wheel, f'{name}.data', 'headers'))
        scripts = str(Path(unpacked_wheel, f'{name}.data', 'scripts'))
        egg_info = dist.reinitialize_command('egg_info', reinit_subcommands=True)
        egg_info.egg_base = str(tmp_dir)
        egg_info.ignore_egg_info_in_manifest = True
        build = dist.reinitialize_command('build', reinit_subcommands=True)
        install = dist.reinitialize_command('install', reinit_subcommands=True)
        build.build_platlib = build.build_purelib = build.build_lib = build_lib
        install.install_purelib = install.install_platlib = install.install_lib = wheel
        install.install_scripts = build.build_scripts = scripts
        install.install_headers = headers
        install.install_data = data
        install_scripts = dist.get_command_obj('install_scripts')
        install_scripts.no_ep = True
        build.build_temp = str(tmp_dir)
        build_py = dist.get_command_obj('build_py')
        build_py.compile = False
        build_py.existing_egg_info_dir = self._find_egg_info_dir()
        self._set_editable_mode()
        build.ensure_finalized()
        install.ensure_finalized()

    def _set_editable_mode(self):
        """Set the ``editable_mode`` flag in the build sub-commands"""
        dist = self.distribution
        build = dist.get_command_obj('build')
        for cmd_name in build.get_sub_commands():
            cmd = dist.get_command_obj(cmd_name)
            if hasattr(cmd, 'editable_mode'):
                cmd.editable_mode = True
            elif hasattr(cmd, 'inplace'):
                cmd.inplace = True

    def _collect_build_outputs(self) -> Tuple[List[str], Dict[str, str]]:
        files: List[str] = []
        mapping: Dict[str, str] = {}
        build = self.get_finalized_command('build')
        for cmd_name in build.get_sub_commands():
            cmd = self.get_finalized_command(cmd_name)
            if hasattr(cmd, 'get_outputs'):
                files.extend(cmd.get_outputs() or [])
            if hasattr(cmd, 'get_output_mapping'):
                mapping.update(cmd.get_output_mapping() or {})
        return (files, mapping)

    def _run_build_commands(self, dist_name: str, unpacked_wheel: _Path, build_lib: _Path, tmp_dir: _Path) -> Tuple[List[str], Dict[str, str]]:
        self._configure_build(dist_name, unpacked_wheel, build_lib, tmp_dir)
        self._run_build_subcommands()
        files, mapping = self._collect_build_outputs()
        self._run_install('headers')
        self._run_install('scripts')
        self._run_install('data')
        return (files, mapping)

    def _run_build_subcommands(self):
        """
        Issue #3501 indicates that some plugins/customizations might rely on:

        1. ``build_py`` not running
        2. ``build_py`` always copying files to ``build_lib``

        However both these assumptions may be false in editable_wheel.
        This method implements a temporary workaround to support the ecosystem
        while the implementations catch up.
        """
        build: Command = self.get_finalized_command('build')
        for name in build.get_sub_commands():
            cmd = self.get_finalized_command(name)
            if name == 'build_py' and type(cmd) != build_py_cls:
                self._safely_run(name)
            else:
                self.run_command(name)

    def _safely_run(self, cmd_name: str):
        try:
            return self.run_command(cmd_name)
        except Exception:
            SetuptoolsDeprecationWarning.emit('Customization incompatible with editable install', f'\n                {traceback.format_exc()}\n\n                If you are seeing this warning it is very likely that a setuptools\n                plugin or customization overrides the `{cmd_name}` command, without\n                taking into consideration how editable installs run build steps\n                starting from setuptools v64.0.0.\n\n                Plugin authors and developers relying on custom build steps are\n                encouraged to update their `{cmd_name}` implementation considering the\n                information about editable installs in\n                https://setuptools.pypa.io/en/latest/userguide/extension.html.\n\n                For the time being `setuptools` will silence this error and ignore\n                the faulty command, but this behaviour will change in future versions.\n                ')

    def _create_wheel_file(self, bdist_wheel):
        from wheel.wheelfile import WheelFile
        dist_info = self.get_finalized_command('dist_info')
        dist_name = dist_info.name
        tag = '-'.join(bdist_wheel.get_tag())
        build_tag = '0.editable'
        archive_name = f'{dist_name}-{build_tag}-{tag}.whl'
        wheel_path = Path(self.dist_dir, archive_name)
        if wheel_path.exists():
            wheel_path.unlink()
        unpacked_wheel = TemporaryDirectory(suffix=archive_name)
        build_lib = TemporaryDirectory(suffix='.build-lib')
        build_tmp = TemporaryDirectory(suffix='.build-temp')
        with unpacked_wheel as unpacked, build_lib as lib, build_tmp as tmp:
            unpacked_dist_info = Path(unpacked, Path(self.dist_info_dir).name)
            shutil.copytree(self.dist_info_dir, unpacked_dist_info)
            self._install_namespaces(unpacked, dist_name)
            files, mapping = self._run_build_commands(dist_name, unpacked, lib, tmp)
            strategy = self._select_strategy(dist_name, tag, lib)
            with strategy, WheelFile(wheel_path, 'w') as wheel_obj:
                strategy(wheel_obj, files, mapping)
                wheel_obj.write_files(unpacked)
        return wheel_path

    def _run_install(self, category: str):
        has_category = getattr(self.distribution, f'has_{category}', None)
        if has_category and has_category():
            _logger.info(f'Installing {category} as non editable')
            self.run_command(f'install_{category}')

    def _select_strategy(self, name: str, tag: str, build_lib: _Path) -> 'EditableStrategy':
        """Decides which strategy to use to implement an editable installation."""
        build_name = f'__editable__.{name}-{tag}'
        project_dir = Path(self.project_dir)
        mode = _EditableMode.convert(self.mode)
        if mode is _EditableMode.STRICT:
            auxiliary_dir = _empty_dir(Path(self.project_dir, 'build', build_name))
            return _LinkTree(self.distribution, name, auxiliary_dir, build_lib)
        packages = _find_packages(self.distribution)
        has_simple_layout = _simple_layout(packages, self.package_dir, project_dir)
        is_compat_mode = mode is _EditableMode.COMPAT
        if set(self.package_dir) == {''} and has_simple_layout or is_compat_mode:
            src_dir = self.package_dir.get('', '.')
            return _StaticPth(self.distribution, name, [Path(project_dir, src_dir)])
        return _TopLevelFinder(self.distribution, name)