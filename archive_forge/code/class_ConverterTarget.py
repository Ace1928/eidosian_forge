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
class ConverterTarget:

    def __init__(self, target: CMakeTarget, env: 'Environment', for_machine: MachineChoice) -> None:
        self.env = env
        self.for_machine = for_machine
        self.artifacts = target.artifacts
        self.src_dir = target.src_dir
        self.build_dir = target.build_dir
        self.name = target.name
        self.cmake_name = target.name
        self.full_name = target.full_name
        self.type = target.type
        self.install = target.install
        self.install_dir: T.Optional[Path] = None
        self.link_libraries = target.link_libraries
        self.link_flags = target.link_flags + target.link_lang_flags
        self.depends_raw: T.List[str] = []
        self.depends: T.List[T.Union[ConverterTarget, ConverterCustomTarget]] = []
        if target.install_paths:
            self.install_dir = target.install_paths[0]
        self.languages: T.Set[str] = set()
        self.sources: T.List[Path] = []
        self.generated: T.List[Path] = []
        self.generated_ctgt: T.List[CustomTargetReference] = []
        self.includes: T.List[Path] = []
        self.sys_includes: T.List[Path] = []
        self.link_with: T.List[T.Union[ConverterTarget, ConverterCustomTarget]] = []
        self.object_libs: T.List[ConverterTarget] = []
        self.compile_opts: T.Dict[str, T.List[str]] = {}
        self.public_compile_opts: T.List[str] = []
        self.pie = False
        self.override_options: T.List[str] = []
        self.name = _sanitize_cmake_name(self.name)
        self.generated_raw: T.List[Path] = []
        for i in target.files:
            languages: T.Set[str] = set()
            src_suffixes: T.Set[str] = set()
            for j in i.sources:
                if not j.suffix:
                    continue
                src_suffixes.add(j.suffix[1:])
            lang_cmake_to_meson = {val.lower(): key for key, val in language_map.items()}
            languages.add(lang_cmake_to_meson.get(i.language.lower(), 'c'))
            for sfx in src_suffixes:
                for key, val in lang_suffixes.items():
                    if sfx in val:
                        languages.add(key)
                        break
            for lang in languages:
                self.languages.add(lang)
                if lang not in self.compile_opts:
                    self.compile_opts[lang] = []
            args = i.flags
            args += [f'-D{x}' for x in i.defines]
            for lang in languages:
                self.compile_opts[lang] += [x for x in args if x not in self.compile_opts[lang]]
            self.includes += [x.path for x in i.includes if x.path not in self.includes and (not x.isSystem)]
            self.sys_includes += [x.path for x in i.includes if x.path not in self.sys_includes and x.isSystem]
            if i.is_generated:
                self.generated_raw += i.sources
            else:
                self.sources += i.sources

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self.name}>'
    std_regex = re.compile('([-]{1,2}std=|/std:v?|[-]{1,2}std:)(.*)')

    def postprocess(self, output_target_map: OutputTargetMap, root_src_dir: Path, subdir: Path, install_prefix: Path, trace: CMakeTraceParser) -> None:
        for i in ['c', 'cpp']:
            if i not in self.compile_opts:
                continue
            temp: T.List[str] = []
            for j in self.compile_opts[i]:
                m = ConverterTarget.std_regex.match(j)
                ctgt = output_target_map.generated(Path(j))
                if m:
                    std = m.group(2)
                    supported = self._all_lang_stds(i)
                    if std not in supported:
                        mlog.warning('Unknown {0}_std "{1}" -> Ignoring. Try setting the project-level {0}_std if build errors occur. Known {0}_stds are: {2}'.format(i, std, ' '.join(supported)), once=True)
                        continue
                    self.override_options += [f'{i}_std={std}']
                elif j in {'-fPIC', '-fpic', '-fPIE', '-fpie'}:
                    self.pie = True
                elif isinstance(ctgt, ConverterCustomTarget):
                    self.generated_raw += [Path(j)]
                    temp += [j]
                elif j in blacklist_compiler_flags:
                    pass
                else:
                    temp += [j]
            self.compile_opts[i] = temp
        if self.type.upper() == 'OBJECT_LIBRARY':
            self.pie = True
        tgt = trace.targets.get(self.cmake_name)
        if tgt:
            self.depends_raw = trace.targets[self.cmake_name].depends
            rtgt = resolve_cmake_trace_targets(self.cmake_name, trace, self.env)
            self.includes += [Path(x) for x in rtgt.include_directories]
            self.link_flags += rtgt.link_flags
            self.public_compile_opts += rtgt.public_compile_opts
            self.link_libraries += rtgt.libraries
        elif self.type.upper() not in ['EXECUTABLE', 'OBJECT_LIBRARY']:
            mlog.warning('CMake: Target', mlog.bold(self.cmake_name), 'not found in CMake trace. This can lead to build errors')
        temp = []
        for i in self.link_libraries:
            if ',-rpath,' in i:
                continue
            if not Path(i).is_absolute():
                link_with = output_target_map.artifact(i)
                if link_with:
                    self.link_with += [link_with]
                    continue
            temp += [i]
        self.link_libraries = temp
        supported = list(assembler_suffixes) + list(header_suffixes) + list(obj_suffixes)
        for i in self.languages:
            supported += list(lang_suffixes[i])
        supported = [f'.{x}' for x in supported]
        self.sources = [x for x in self.sources if any((x.name.endswith(y) for y in supported))]
        self.generated_raw = [x for x in self.generated_raw if not x.name.endswith('.rule')]

        def rel_path(x: Path, is_header: bool, is_generated: bool) -> T.Optional[Path]:
            if not x.is_absolute():
                x = self.src_dir / x
            x = x.resolve()
            assert x.is_absolute()
            if not x.exists() and (not any((x.name.endswith(y) for y in obj_suffixes))) and (not is_generated):
                if path_is_in_root(x, Path(self.env.get_build_dir()), resolve=True):
                    x.mkdir(parents=True, exist_ok=True)
                    return x.relative_to(Path(self.env.get_build_dir()) / subdir)
                else:
                    mlog.warning('CMake: path', mlog.bold(x.as_posix()), 'does not exist.')
                    mlog.warning(' --> Ignoring. This can lead to build errors.')
                    return None
            if x in trace.explicit_headers:
                return None
            if path_is_in_root(x, Path(self.env.get_source_dir())) and (not (path_is_in_root(x, root_src_dir) or path_is_in_root(x, Path(self.env.get_build_dir())))):
                mlog.warning('CMake: path', mlog.bold(x.as_posix()), 'is inside the root project but', mlog.bold('not'), 'inside the subproject.')
                mlog.warning(' --> Ignoring. This can lead to build errors.')
                return None
            if path_is_in_root(x, Path(self.env.get_build_dir())) and is_header:
                return x.relative_to(Path(self.env.get_build_dir()) / subdir)
            if path_is_in_root(x, root_src_dir):
                return x.relative_to(root_src_dir)
            return x
        build_dir_rel = self.build_dir.relative_to(Path(self.env.get_build_dir()) / subdir)
        self.generated_raw = [rel_path(x, False, True) for x in self.generated_raw]
        self.includes = list(OrderedSet([rel_path(x, True, False) for x in OrderedSet(self.includes)] + [build_dir_rel]))
        self.sys_includes = list(OrderedSet([rel_path(x, True, False) for x in OrderedSet(self.sys_includes)]))
        self.sources = [rel_path(x, False, False) for x in self.sources]
        for gen_file in self.generated_raw:
            ctgt = output_target_map.generated(gen_file)
            if ctgt:
                assert isinstance(ctgt, ConverterCustomTarget)
                ref = ctgt.get_ref(gen_file)
                assert isinstance(ref, CustomTargetReference) and ref.valid()
                self.generated_ctgt += [ref]
            elif gen_file is not None:
                self.generated += [gen_file]
        self.includes = [x for x in self.includes if x is not None]
        self.sys_includes = [x for x in self.sys_includes if x is not None]
        self.sources = [x for x in self.sources if x is not None]
        if Path('.') not in self.includes:
            self.includes += [Path('.')]
        if self.install_dir and self.install_dir.is_absolute():
            if path_is_in_root(self.install_dir, install_prefix):
                self.install_dir = self.install_dir.relative_to(install_prefix)

        def check_flag(flag: str) -> bool:
            if flag.lower() in blacklist_link_flags or flag in blacklist_compiler_flags + blacklist_clang_cl_link_flags:
                return False
            if flag.startswith('/D'):
                return False
            return True
        self.link_libraries = [x for x in self.link_libraries if x.lower() not in blacklist_link_libs]
        self.link_flags = [x for x in self.link_flags if check_flag(x)]

        def handle_frameworks(flags: T.List[str]) -> T.List[str]:
            res: T.List[str] = []
            for i in flags:
                p = Path(i)
                if not p.exists() or not p.name.endswith('.framework'):
                    res += [i]
                    continue
                res += ['-framework', p.stem]
            return res
        self.link_libraries = handle_frameworks(self.link_libraries)
        self.link_flags = handle_frameworks(self.link_flags)
        for i in self.depends_raw:
            dep_tgt = output_target_map.target(i)
            if dep_tgt:
                self.depends.append(dep_tgt)

    def process_object_libs(self, obj_target_list: T.List['ConverterTarget'], linker_workaround: bool) -> None:
        temp = [x for x in self.generated if any((x.name.endswith('.' + y) for y in obj_suffixes))]
        stem = [x.stem for x in temp]
        exts = self._all_source_suffixes()
        for i in obj_target_list:
            source_files = [x.name for x in i.sources + i.generated]
            for j in stem:
                candidates = [j]
                if not any((j.endswith('.' + x) for x in exts)):
                    mlog.warning('Object files do not contain source file extensions, thus falling back to guessing them.', once=True)
                    candidates += [f'{j}.{x}' for x in exts]
                if any((x in source_files for x in candidates)):
                    if linker_workaround:
                        self._append_objlib_sources(i)
                    else:
                        self.includes += i.includes
                        self.includes = list(OrderedSet(self.includes))
                        self.object_libs += [i]
                    break
        self.generated = [x for x in self.generated if not any((x.name.endswith('.' + y) for y in obj_suffixes))]

    def _append_objlib_sources(self, tgt: 'ConverterTarget') -> None:
        self.includes += tgt.includes
        self.sources += tgt.sources
        self.generated += tgt.generated
        self.generated_ctgt += tgt.generated_ctgt
        self.includes = list(OrderedSet(self.includes))
        self.sources = list(OrderedSet(self.sources))
        self.generated = list(OrderedSet(self.generated))
        self.generated_ctgt = list(OrderedSet(self.generated_ctgt))
        for lang, opts in tgt.compile_opts.items():
            if lang not in self.compile_opts:
                self.compile_opts[lang] = []
            self.compile_opts[lang] += [x for x in opts if x not in self.compile_opts[lang]]

    @lru_cache(maxsize=None)
    def _all_source_suffixes(self) -> 'ImmutableListProtocol[str]':
        suffixes: T.List[str] = []
        for exts in lang_suffixes.values():
            suffixes.extend(exts)
        return suffixes

    @lru_cache(maxsize=None)
    def _all_lang_stds(self, lang: str) -> 'ImmutableListProtocol[str]':
        try:
            res = self.env.coredata.options[OptionKey('std', machine=MachineChoice.BUILD, lang=lang)].choices
        except KeyError:
            return []
        assert isinstance(res, list)
        for i in res:
            assert isinstance(i, str)
        return res

    def process_inter_target_dependencies(self) -> None:
        to_process = list(self.depends)
        processed = []
        new_deps = []
        for i in to_process:
            processed += [i]
            if isinstance(i, ConverterTarget) and i.meson_func() in transfer_dependencies_from:
                to_process += [x for x in i.depends if x not in processed]
            else:
                new_deps += [i]
        self.depends = list(OrderedSet(new_deps))

    def cleanup_dependencies(self) -> None:
        if self.meson_func() in transfer_dependencies_from:
            self.depends = []

    def meson_func(self) -> str:
        return target_type_map.get(self.type.upper())

    def log(self) -> None:
        mlog.log('Target', mlog.bold(self.name), f'({self.cmake_name})')
        mlog.log('  -- artifacts:      ', mlog.bold(str(self.artifacts)))
        mlog.log('  -- full_name:      ', mlog.bold(self.full_name))
        mlog.log('  -- type:           ', mlog.bold(self.type))
        mlog.log('  -- install:        ', mlog.bold('true' if self.install else 'false'))
        mlog.log('  -- install_dir:    ', mlog.bold(self.install_dir.as_posix() if self.install_dir else ''))
        mlog.log('  -- link_libraries: ', mlog.bold(str(self.link_libraries)))
        mlog.log('  -- link_with:      ', mlog.bold(str(self.link_with)))
        mlog.log('  -- object_libs:    ', mlog.bold(str(self.object_libs)))
        mlog.log('  -- link_flags:     ', mlog.bold(str(self.link_flags)))
        mlog.log('  -- languages:      ', mlog.bold(str(self.languages)))
        mlog.log('  -- includes:       ', mlog.bold(str(self.includes)))
        mlog.log('  -- sys_includes:   ', mlog.bold(str(self.sys_includes)))
        mlog.log('  -- sources:        ', mlog.bold(str(self.sources)))
        mlog.log('  -- generated:      ', mlog.bold(str(self.generated)))
        mlog.log('  -- generated_ctgt: ', mlog.bold(str(self.generated_ctgt)))
        mlog.log('  -- pie:            ', mlog.bold('true' if self.pie else 'false'))
        mlog.log('  -- override_opts:  ', mlog.bold(str(self.override_options)))
        mlog.log('  -- depends:        ', mlog.bold(str(self.depends)))
        mlog.log('  -- options:')
        for key, val in self.compile_opts.items():
            mlog.log('    -', key, '=', mlog.bold(str(val)))