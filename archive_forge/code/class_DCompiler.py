from __future__ import annotations
import os.path
import re
import subprocess
import typing as T
from .. import mesonlib
from .. import mlog
from ..arglist import CompilerArgs
from ..linkers import RSPFileSyntax
from ..mesonlib import (
from . import compilers
from .compilers import (
from .mixins.gnu import GnuCompiler
from .mixins.gnu import gnu_common_warning_args
class DCompiler(Compiler):
    mscrt_args = {'none': ['-mscrtlib='], 'md': ['-mscrtlib=msvcrt'], 'mdd': ['-mscrtlib=msvcrtd'], 'mt': ['-mscrtlib=libcmt'], 'mtd': ['-mscrtlib=libcmtd']}
    language = 'd'

    def __init__(self, exelist: T.List[str], version: str, for_machine: MachineChoice, info: 'MachineInfo', arch: str, *, exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None, is_cross: bool=False):
        super().__init__([], exelist, version, for_machine, info, linker=linker, full_version=full_version, is_cross=is_cross)
        self.arch = arch
        self.exe_wrapper = exe_wrapper

    def sanity_check(self, work_dir: str, environment: 'Environment') -> None:
        source_name = os.path.join(work_dir, 'sanity.d')
        output_name = os.path.join(work_dir, 'dtest')
        with open(source_name, 'w', encoding='utf-8') as ofile:
            ofile.write('void main() { }')
        pc = subprocess.Popen(self.exelist + self.get_output_args(output_name) + self._get_target_arch_args() + [source_name], cwd=work_dir)
        pc.wait()
        if pc.returncode != 0:
            raise EnvironmentException('D compiler %s cannot compile programs.' % self.name_string())
        if self.is_cross:
            if self.exe_wrapper is None:
                return
            cmdlist = self.exe_wrapper.get_command() + [output_name]
        else:
            cmdlist = [output_name]
        if subprocess.call(cmdlist) != 0:
            raise EnvironmentException('Executables created by D compiler %s are not runnable.' % self.name_string())

    def needs_static_linker(self) -> bool:
        return True

    def get_depfile_suffix(self) -> str:
        return 'deps'

    def get_pic_args(self) -> T.List[str]:
        if self.info.is_windows():
            return []
        return ['-fPIC']

    def get_feature_args(self, kwargs: DFeatures, build_to_src: str) -> T.List[str]:
        res: T.List[str] = []
        unittest_arg = d_feature_args[self.id]['unittest']
        if not unittest_arg:
            raise EnvironmentException('D compiler %s does not support the "unittest" feature.' % self.name_string())
        if kwargs['unittest']:
            res.append(unittest_arg)
        debug_level = -1
        debug_arg = d_feature_args[self.id]['debug']
        if not debug_arg:
            raise EnvironmentException('D compiler %s does not support conditional debug identifiers.' % self.name_string())
        for d in kwargs['debug']:
            if isinstance(d, int):
                debug_level = max(debug_level, d)
            elif isinstance(d, str) and d.isdigit():
                debug_level = max(debug_level, int(d))
            else:
                res.append(f'{debug_arg}={d}')
        if debug_level >= 0:
            res.append(f'{debug_arg}={debug_level}')
        version_level = -1
        version_arg = d_feature_args[self.id]['version']
        if not version_arg:
            raise EnvironmentException('D compiler %s does not support conditional version identifiers.' % self.name_string())
        for v in kwargs['versions']:
            if isinstance(v, int):
                version_level = max(version_level, v)
            elif isinstance(v, str) and v.isdigit():
                version_level = max(version_level, int(v))
            else:
                res.append(f'{version_arg}={v}')
        if version_level >= 0:
            res.append(f'{version_arg}={version_level}')
        import_dir_arg = d_feature_args[self.id]['import_dir']
        if not import_dir_arg:
            raise EnvironmentException('D compiler %s does not support the "string import directories" feature.' % self.name_string())
        for idir_obj in kwargs['import_dirs']:
            basedir = idir_obj.get_curdir()
            for idir in idir_obj.get_incdirs():
                bldtreedir = os.path.join(basedir, idir)
                if idir not in ('', '.'):
                    expdir = bldtreedir
                else:
                    expdir = basedir
                srctreedir = os.path.join(build_to_src, expdir)
                res.append(f'{import_dir_arg}{srctreedir}')
                res.append(f'{import_dir_arg}{bldtreedir}')
        return res

    def get_optimization_link_args(self, optimization_level: str) -> T.List[str]:
        if optimization_level != 'plain':
            return self._get_target_arch_args()
        return []

    def compiler_args(self, args: T.Optional[T.Iterable[str]]=None) -> DCompilerArgs:
        return DCompilerArgs(self, args)

    def has_multi_arguments(self, args: T.List[str], env: 'Environment') -> T.Tuple[bool, bool]:
        return self.compiles('int i;\n', env, extra_args=args)

    def _get_target_arch_args(self) -> T.List[str]:
        if self.info.is_windows():
            if self.arch == 'x86_64':
                return ['-m64']
            return ['-m32']
        return []

    def get_crt_compile_args(self, crt_val: str, buildtype: str) -> T.List[str]:
        return []

    def get_crt_link_args(self, crt_val: str, buildtype: str) -> T.List[str]:
        return []

    def _get_compile_extra_args(self, extra_args: T.Union[T.List[str], T.Callable[[CompileCheckMode], T.List[str]], None]=None) -> T.List[str]:
        args = self._get_target_arch_args()
        if extra_args:
            if callable(extra_args):
                extra_args = extra_args(CompileCheckMode.COMPILE)
            if isinstance(extra_args, list):
                args.extend(extra_args)
            elif isinstance(extra_args, str):
                args.append(extra_args)
        return args

    def run(self, code: 'mesonlib.FileOrString', env: 'Environment', *, extra_args: T.Union[T.List[str], T.Callable[[CompileCheckMode], T.List[str]], None]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> compilers.RunResult:
        need_exe_wrapper = env.need_exe_wrapper(self.for_machine)
        if need_exe_wrapper and self.exe_wrapper is None:
            raise compilers.CrossNoRunException('Can not run test applications in this cross environment.')
        extra_args = self._get_compile_extra_args(extra_args)
        with self._build_wrapper(code, env, extra_args, dependencies, mode=CompileCheckMode.LINK, want_output=True) as p:
            if p.returncode != 0:
                mlog.debug(f'Could not compile test file {p.input_name}: {p.returncode}\n')
                return compilers.RunResult(False)
            if need_exe_wrapper:
                cmdlist = self.exe_wrapper.get_command() + [p.output_name]
            else:
                cmdlist = [p.output_name]
            try:
                pe, so, se = mesonlib.Popen_safe(cmdlist)
            except Exception as e:
                mlog.debug(f'Could not run: {cmdlist} (error: {e})\n')
                return compilers.RunResult(False)
        mlog.debug('Program stdout:\n')
        mlog.debug(so)
        mlog.debug('Program stderr:\n')
        mlog.debug(se)
        return compilers.RunResult(True, pe.returncode, so, se)

    def sizeof(self, typename: str, prefix: str, env: 'Environment', *, extra_args: T.Union[None, T.List[str], T.Callable[[CompileCheckMode], T.List[str]]]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> T.Tuple[int, bool]:
        if extra_args is None:
            extra_args = []
        t = f'\n        import std.stdio : writeln;\n        {prefix}\n        void main() {{\n            writeln(({typename}).sizeof);\n        }}\n        '
        res = self.cached_run(t, env, extra_args=extra_args, dependencies=dependencies)
        if not res.compiled:
            return (-1, False)
        if res.returncode != 0:
            raise mesonlib.EnvironmentException('Could not run sizeof test binary.')
        return (int(res.stdout), res.cached)

    def alignment(self, typename: str, prefix: str, env: 'Environment', *, extra_args: T.Optional[T.List[str]]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> T.Tuple[int, bool]:
        if extra_args is None:
            extra_args = []
        t = f'\n        import std.stdio : writeln;\n        {prefix}\n        void main() {{\n            writeln(({typename}).alignof);\n        }}\n        '
        res = self.run(t, env, extra_args=extra_args, dependencies=dependencies)
        if not res.compiled:
            raise mesonlib.EnvironmentException('Could not compile alignment test.')
        if res.returncode != 0:
            raise mesonlib.EnvironmentException('Could not run alignment test binary.')
        align = int(res.stdout)
        if align == 0:
            raise mesonlib.EnvironmentException(f'Could not determine alignment of {typename}. Sorry. You might want to file a bug.')
        return (align, res.cached)

    def has_header(self, hname: str, prefix: str, env: 'Environment', *, extra_args: T.Union[None, T.List[str], T.Callable[['CompileCheckMode'], T.List[str]]]=None, dependencies: T.Optional[T.List['Dependency']]=None, disable_cache: bool=False) -> T.Tuple[bool, bool]:
        extra_args = self._get_compile_extra_args(extra_args)
        code = f'{prefix}\n        import {hname};\n        '
        return self.compiles(code, env, extra_args=extra_args, dependencies=dependencies, mode=CompileCheckMode.COMPILE, disable_cache=disable_cache)