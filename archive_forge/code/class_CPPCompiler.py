from __future__ import annotations
import copy
import functools
import os.path
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import MesonException, version_compare, OptionKey
from .compilers import (
from .c_function_attributes import CXX_FUNC_ATTRIBUTES, C_FUNC_ATTRIBUTES
from .mixins.clike import CLikeCompiler
from .mixins.ccrx import CcrxCompiler
from .mixins.ti import TICompiler
from .mixins.arm import ArmCompiler, ArmclangCompiler
from .mixins.visualstudio import MSVCCompiler, ClangClCompiler
from .mixins.gnu import GnuCompiler, gnu_common_warning_args, gnu_cpp_warning_args
from .mixins.intel import IntelGnuLikeCompiler, IntelVisualStudioLikeCompiler
from .mixins.clang import ClangCompiler
from .mixins.elbrus import ElbrusCompiler
from .mixins.pgi import PGICompiler
from .mixins.emscripten import EmscriptenMixin
from .mixins.metrowerks import MetrowerksCompiler
from .mixins.metrowerks import mwccarm_instruction_set_args, mwcceppc_instruction_set_args
class CPPCompiler(CLikeCompiler, Compiler):

    def attribute_check_func(self, name: str) -> str:
        try:
            return CXX_FUNC_ATTRIBUTES.get(name, C_FUNC_ATTRIBUTES[name])
        except KeyError:
            raise MesonException(f'Unknown function attribute "{name}"')
    language = 'cpp'

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None):
        Compiler.__init__(self, ccache, exelist, version, for_machine, info, is_cross=is_cross, linker=linker, full_version=full_version)
        CLikeCompiler.__init__(self, exe_wrapper)

    @classmethod
    def get_display_language(cls) -> str:
        return 'C++'

    def get_no_stdinc_args(self) -> T.List[str]:
        return ['-nostdinc++']

    def get_no_stdlib_link_args(self) -> T.List[str]:
        return ['-nostdlib++']

    def sanity_check(self, work_dir: str, environment: 'Environment') -> None:
        code = 'class breakCCompiler;int main(void) { return 0; }\n'
        return self._sanity_check_impl(work_dir, environment, 'sanitycheckcpp.cc', code)

    def get_compiler_check_args(self, mode: CompileCheckMode) -> T.List[str]:
        return super().get_compiler_check_args(mode) + ['-fpermissive']

    def has_header_symbol(self, hname: str, symbol: str, prefix: str, env: 'Environment', *, extra_args: T.Union[None, T.List[str], T.Callable[[CompileCheckMode], T.List[str]]]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> T.Tuple[bool, bool]:
        found, cached = super().has_header_symbol(hname, symbol, prefix, env, extra_args=extra_args, dependencies=dependencies)
        if found:
            return (True, cached)
        if extra_args is None:
            extra_args = []
        t = f'{prefix}\n        #include <{hname}>\n        using {symbol};\n        int main(void) {{ return 0; }}'
        return self.compiles(t, env, extra_args=extra_args, dependencies=dependencies)

    def _test_cpp_std_arg(self, cpp_std_value: str) -> bool:
        assert cpp_std_value.startswith('-std=')
        CPP_TEST = 'int i = static_cast<int>(0);'
        with self.compile(CPP_TEST, extra_args=[cpp_std_value], mode=CompileCheckMode.COMPILE) as p:
            if p.returncode == 0:
                mlog.debug(f'Compiler accepts {cpp_std_value}:', 'YES')
                return True
            else:
                mlog.debug(f'Compiler accepts {cpp_std_value}:', 'NO')
                return False

    @functools.lru_cache()
    def _find_best_cpp_std(self, cpp_std: str) -> str:
        CPP_FALLBACKS = {'c++11': 'c++0x', 'gnu++11': 'gnu++0x', 'c++14': 'c++1y', 'gnu++14': 'gnu++1y', 'c++17': 'c++1z', 'gnu++17': 'gnu++1z', 'c++20': 'c++2a', 'gnu++20': 'gnu++2a', 'c++23': 'c++2b', 'gnu++23': 'gnu++2b', 'c++26': 'c++2c', 'gnu++26': 'gnu++2c'}
        assert self.id in frozenset(['clang', 'lcc', 'gcc', 'emscripten', 'armltdclang', 'intel-llvm'])
        if cpp_std not in CPP_FALLBACKS:
            return '-std=' + cpp_std
        for i in (cpp_std, CPP_FALLBACKS[cpp_std]):
            cpp_std_value = '-std=' + i
            if self._test_cpp_std_arg(cpp_std_value):
                return cpp_std_value
        raise MesonException(f'C++ Compiler does not support -std={cpp_std}')

    def get_options(self) -> 'MutableKeyedOptionDictType':
        opts = super().get_options()
        key = OptionKey('std', machine=self.for_machine, lang=self.language)
        opts.update({key: coredata.UserStdOption('C++', _ALL_STDS)})
        return opts