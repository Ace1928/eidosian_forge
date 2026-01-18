from __future__ import annotations
import os
import errno
import shutil
import subprocess
import sys
from pathlib import Path
from ._backend import Backend
from string import Template
from itertools import chain
import warnings
class MesonBackend(Backend):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dependencies = self.extra_dat.get('dependencies', [])
        self.meson_build_dir = 'bbdir'
        self.build_type = 'debug' if any(('debug' in flag for flag in self.fc_flags)) else 'release'

    def _move_exec_to_root(self, build_dir: Path):
        walk_dir = Path(build_dir) / self.meson_build_dir
        path_objects = chain(walk_dir.glob(f'{self.modulename}*.so'), walk_dir.glob(f'{self.modulename}*.pyd'))
        for path_object in path_objects:
            dest_path = Path.cwd() / path_object.name
            if dest_path.exists():
                dest_path.unlink()
            shutil.copy2(path_object, dest_path)
            os.remove(path_object)

    def write_meson_build(self, build_dir: Path) -> None:
        """Writes the meson build file at specified location"""
        meson_template = MesonTemplate(self.modulename, self.sources, self.dependencies, self.libraries, self.library_dirs, self.include_dirs, self.extra_objects, self.flib_flags, self.fc_flags, self.build_type, sys.executable)
        src = meson_template.generate_meson_build()
        Path(build_dir).mkdir(parents=True, exist_ok=True)
        meson_build_file = Path(build_dir) / 'meson.build'
        meson_build_file.write_text(src)
        return meson_build_file

    def _run_subprocess_command(self, command, cwd):
        subprocess.run(command, cwd=cwd, check=True)

    def run_meson(self, build_dir: Path):
        setup_command = ['meson', 'setup', self.meson_build_dir]
        self._run_subprocess_command(setup_command, build_dir)
        compile_command = ['meson', 'compile', '-C', self.meson_build_dir]
        self._run_subprocess_command(compile_command, build_dir)

    def compile(self) -> None:
        self.sources = _prepare_sources(self.modulename, self.sources, self.build_dir)
        self.write_meson_build(self.build_dir)
        self.run_meson(self.build_dir)
        self._move_exec_to_root(self.build_dir)