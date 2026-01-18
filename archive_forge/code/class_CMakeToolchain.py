from __future__ import annotations
from pathlib import Path
from .traceparser import CMakeTraceParser
from ..envconfig import CMakeSkipCompilerTest
from .common import language_map, cmake_get_generator_args
from .. import mlog
import shutil
import typing as T
from enum import Enum
from textwrap import dedent
class CMakeToolchain:

    def __init__(self, cmakebin: 'CMakeExecutor', env: 'Environment', for_machine: MachineChoice, exec_scope: CMakeExecScope, build_dir: Path, preload_file: T.Optional[Path]=None) -> None:
        self.env = env
        self.cmakebin = cmakebin
        self.for_machine = for_machine
        self.exec_scope = exec_scope
        self.preload_file = preload_file
        self.build_dir = build_dir
        self.build_dir = self.build_dir.resolve()
        self.toolchain_file = build_dir / 'CMakeMesonToolchainFile.cmake'
        self.cmcache_file = build_dir / 'CMakeCache.txt'
        self.minfo = self.env.machines[self.for_machine]
        self.properties = self.env.properties[self.for_machine]
        self.compilers = self.env.coredata.compilers[self.for_machine]
        self.cmakevars = self.env.cmakevars[self.for_machine]
        self.cmakestate = self.env.coredata.cmake_cache[self.for_machine]
        self.variables = self.get_defaults()
        self.variables.update(self.cmakevars.get_variables())
        skip_status = self.properties.get_cmake_skip_compiler_test()
        self.skip_check = skip_status == CMakeSkipCompilerTest.ALWAYS
        if skip_status == CMakeSkipCompilerTest.DEP_ONLY and self.exec_scope == CMakeExecScope.DEPENDENCY:
            self.skip_check = True
        if not self.properties.get_cmake_defaults():
            self.skip_check = False
        assert self.toolchain_file.is_absolute()

    def write(self) -> Path:
        if not self.toolchain_file.parent.exists():
            self.toolchain_file.parent.mkdir(parents=True)
        self.toolchain_file.write_text(self.generate(), encoding='utf-8')
        self.cmcache_file.write_text(self.generate_cache(), encoding='utf-8')
        mlog.cmd_ci_include(self.toolchain_file.as_posix())
        return self.toolchain_file

    def get_cmake_args(self) -> T.List[str]:
        args = ['-DCMAKE_TOOLCHAIN_FILE=' + self.toolchain_file.as_posix()]
        if self.preload_file is not None:
            args += ['-DMESON_PRELOAD_FILE=' + self.preload_file.as_posix()]
        return args

    @staticmethod
    def _print_vars(vars: T.Dict[str, T.List[str]]) -> str:
        res = ''
        for key, value in vars.items():
            res += 'set(' + key
            for i in value:
                res += f' "{i}"'
            res += ')\n'
        return res

    def generate(self) -> str:
        res = dedent('            ######################################\n            ###  AUTOMATICALLY GENERATED FILE  ###\n            ######################################\n\n            # This file was generated from the configuration in the\n            # relevant meson machine file. See the meson documentation\n            # https://mesonbuild.com/Machine-files.html for more information\n\n            if(DEFINED MESON_PRELOAD_FILE)\n                include("${MESON_PRELOAD_FILE}")\n            endif()\n\n        ')
        for key, value in self.variables.items():
            self.variables[key] = [x.replace('\\', '/') for x in value]
        if self.skip_check:
            self.update_cmake_compiler_state()
            res += '# CMake compiler state variables\n'
            for lang, vars in self.cmakestate:
                res += f'# -- Variables for language {lang}\n'
                res += self._print_vars(vars)
                res += '\n'
            res += '\n'
        res += '# Variables from meson\n'
        res += self._print_vars(self.variables)
        res += '\n'
        user_file = self.properties.get_cmake_toolchain_file()
        if user_file is not None:
            res += dedent('\n                # Load the CMake toolchain file specified by the user\n                include("{}")\n\n            '.format(user_file.as_posix()))
        return res

    def generate_cache(self) -> str:
        if not self.skip_check:
            return ''
        res = ''
        for name, v in self.cmakestate.cmake_cache.items():
            res += f'{name}:{v.type}={';'.join(v.value)}\n'
        return res

    def get_defaults(self) -> T.Dict[str, T.List[str]]:
        defaults: T.Dict[str, T.List[str]] = {}
        if not self.properties.get_cmake_defaults():
            return defaults
        SYSTEM_MAP: T.Dict[str, str] = {'android': 'Android', 'linux': 'Linux', 'windows': 'Windows', 'freebsd': 'FreeBSD', 'darwin': 'Darwin'}
        if self.env.is_cross_build(when_building_for=self.for_machine):
            defaults['CMAKE_SYSTEM_NAME'] = [SYSTEM_MAP.get(self.minfo.system, self.minfo.system)]
            defaults['CMAKE_SYSTEM_PROCESSOR'] = [self.minfo.cpu_family]
        defaults['CMAKE_SIZEOF_VOID_P'] = ['8' if self.minfo.is_64_bit else '4']
        sys_root = self.properties.get_sys_root()
        if sys_root:
            defaults['CMAKE_SYSROOT'] = [sys_root]

        def make_abs(exe: str) -> str:
            if Path(exe).is_absolute():
                return exe
            p = shutil.which(exe)
            if p is None:
                return exe
            return p
        for lang, comp_obj in self.compilers.items():
            prefix = 'CMAKE_{}_'.format(language_map.get(lang, lang.upper()))
            exe_list = comp_obj.get_exelist()
            if not exe_list:
                continue
            if len(exe_list) >= 2 and (not self.is_cmdline_option(comp_obj, exe_list[1])):
                defaults[prefix + 'COMPILER_LAUNCHER'] = [make_abs(exe_list[0])]
                exe_list = exe_list[1:]
            exe_list[0] = make_abs(exe_list[0])
            defaults[prefix + 'COMPILER'] = exe_list
            if comp_obj.get_id() == 'clang-cl':
                defaults['CMAKE_LINKER'] = comp_obj.get_linker_exelist()
        return defaults

    @staticmethod
    def is_cmdline_option(compiler: 'Compiler', arg: str) -> bool:
        if compiler.get_argument_syntax() == 'msvc':
            return arg.startswith('/')
        else:
            return arg.startswith('-')

    def update_cmake_compiler_state(self) -> None:
        if self.cmakestate.languages.issuperset(self.compilers.keys()):
            return
        mlog.debug('CMake Toolchain: Calling CMake once to generate the compiler state')
        languages = list(self.compilers.keys())
        lang_ids = [language_map.get(x, x.upper()) for x in languages]
        cmake_content = dedent(f'\n            cmake_minimum_required(VERSION 3.7)\n            project(CompInfo {' '.join(lang_ids)})\n        ')
        build_dir = Path(self.env.scratch_dir) / '__CMake_compiler_info__'
        build_dir.mkdir(parents=True, exist_ok=True)
        cmake_file = build_dir / 'CMakeLists.txt'
        cmake_file.write_text(cmake_content, encoding='utf-8')
        temp_toolchain_file = build_dir / 'CMakeMesonTempToolchainFile.cmake'
        temp_toolchain_file.write_text(CMakeToolchain._print_vars(self.variables), encoding='utf-8')
        trace = CMakeTraceParser(self.cmakebin.version(), build_dir, self.env)
        self.cmakebin.set_exec_mode(print_cmout=False, always_capture_stderr=trace.requires_stderr())
        cmake_args = []
        cmake_args += trace.trace_args()
        cmake_args += cmake_get_generator_args(self.env)
        cmake_args += [f'-DCMAKE_TOOLCHAIN_FILE={temp_toolchain_file.as_posix()}', '.']
        rc, _, raw_trace = self.cmakebin.call(cmake_args, build_dir=build_dir, disable_cache=True)
        if rc != 0:
            mlog.warning('CMake Toolchain: Failed to determine CMake compilers state')
            return
        trace.parse(raw_trace)
        self.cmakestate.cmake_cache = {**trace.cache}
        vars_by_file = {k.name: v for k, v in trace.vars_by_file.items()}
        for lang in languages:
            lang_cmake = language_map.get(lang, lang.upper())
            file_name = f'CMake{lang_cmake}Compiler.cmake'
            vars = vars_by_file.setdefault(file_name, {})
            vars[f'CMAKE_{lang_cmake}_COMPILER_FORCED'] = ['1']
            self.cmakestate.update(lang, vars)