from __future__ import annotations
from functools import lru_cache
from os import environ
from pathlib import Path
import re
import typing as T
from .common import CMakeException, CMakeTarget, language_map, cmake_get_generator_args, check_cmake_args
from .fileapi import CMakeFileAPI
from .executor import CMakeExecutor
from .toolchain import CMakeToolchain, CMakeExecScope
from .traceparser import CMakeTraceParser
from .tracetargets import resolve_cmake_trace_targets
from .. import mlog, mesonlib
from ..mesonlib import MachineChoice, OrderedSet, path_is_in_root, relative_to_if_possible, OptionKey
from ..mesondata import DataFile
from ..compilers.compilers import assembler_suffixes, lang_suffixes, header_suffixes, obj_suffixes, lib_suffixes, is_header
from ..programs import ExternalProgram
from ..coredata import FORBIDDEN_TARGET_NAMES
from ..mparser import (
class OutputTargetMap:
    rm_so_version = re.compile('(\\.[0-9]+)+$')

    def __init__(self, build_dir: Path):
        self.tgt_map: T.Dict[str, T.Union['ConverterTarget', 'ConverterCustomTarget']] = {}
        self.build_dir = build_dir

    def add(self, tgt: T.Union['ConverterTarget', 'ConverterCustomTarget']) -> None:

        def assign_keys(keys: T.List[str]) -> None:
            for i in [x for x in keys if x]:
                self.tgt_map[i] = tgt
        keys = [self._target_key(tgt.cmake_name)]
        if isinstance(tgt, ConverterTarget):
            keys += [tgt.full_name]
            keys += [self._rel_artifact_key(x) for x in tgt.artifacts]
            keys += [self._base_artifact_key(x) for x in tgt.artifacts]
        if isinstance(tgt, ConverterCustomTarget):
            keys += [self._rel_generated_file_key(x) for x in tgt.original_outputs]
            keys += [self._base_generated_file_key(x) for x in tgt.original_outputs]
        assign_keys(keys)

    def _return_first_valid_key(self, keys: T.List[str]) -> T.Optional[T.Union['ConverterTarget', 'ConverterCustomTarget']]:
        for i in keys:
            if i and i in self.tgt_map:
                return self.tgt_map[i]
        return None

    def target(self, name: str) -> T.Optional[T.Union['ConverterTarget', 'ConverterCustomTarget']]:
        return self._return_first_valid_key([self._target_key(name)])

    def executable(self, name: str) -> T.Optional['ConverterTarget']:
        tgt = self.target(name)
        if tgt is None or not isinstance(tgt, ConverterTarget):
            return None
        if tgt.meson_func() != 'executable':
            return None
        return tgt

    def artifact(self, name: str) -> T.Optional[T.Union['ConverterTarget', 'ConverterCustomTarget']]:
        keys = []
        candidates = [name, OutputTargetMap.rm_so_version.sub('', name)]
        for i in lib_suffixes:
            if not name.endswith('.' + i):
                continue
            new_name = name[:-len(i) - 1]
            new_name = OutputTargetMap.rm_so_version.sub('', new_name)
            candidates += [f'{new_name}.{i}']
        for i in candidates:
            keys += [self._rel_artifact_key(Path(i)), Path(i).name, self._base_artifact_key(Path(i))]
        return self._return_first_valid_key(keys)

    def generated(self, name: Path) -> T.Optional['ConverterCustomTarget']:
        res = self._return_first_valid_key([self._rel_generated_file_key(name), self._base_generated_file_key(name)])
        assert res is None or isinstance(res, ConverterCustomTarget)
        return res

    def _rel_path(self, fname: Path) -> T.Optional[Path]:
        try:
            return fname.resolve().relative_to(self.build_dir)
        except ValueError:
            pass
        return None

    def _target_key(self, tgt_name: str) -> str:
        return f'__tgt_{tgt_name}__'

    def _rel_generated_file_key(self, fname: Path) -> T.Optional[str]:
        path = self._rel_path(fname)
        return f'__relgen_{path.as_posix()}__' if path else None

    def _base_generated_file_key(self, fname: Path) -> str:
        return f'__gen_{fname.name}__'

    def _rel_artifact_key(self, fname: Path) -> T.Optional[str]:
        path = self._rel_path(fname)
        return f'__relart_{path.as_posix()}__' if path else None

    def _base_artifact_key(self, fname: Path) -> str:
        return f'__art_{fname.name}__'