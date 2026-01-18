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
class DmdLikeCompilerMixin(CompilerMixinBase):
    """Mixin class for DMD and LDC.

    LDC has a number of DMD like arguments, and this class allows for code
    sharing between them as makes sense.
    """

    def __init__(self, dmd_frontend_version: T.Optional[str]):
        if dmd_frontend_version is None:
            self._dmd_has_depfile = False
        else:
            self._dmd_has_depfile = version_compare(dmd_frontend_version, '>=2.095.0')
    if T.TYPE_CHECKING:
        mscrt_args: T.Dict[str, T.List[str]] = {}

        def _get_target_arch_args(self) -> T.List[str]:
            ...
    LINKER_PREFIX = '-L='

    def get_output_args(self, outputname: str) -> T.List[str]:
        return ['-of=' + outputname]

    def get_linker_output_args(self, outputname: str) -> T.List[str]:
        return ['-of=' + outputname]

    def get_include_args(self, path: str, is_system: bool) -> T.List[str]:
        if path == '':
            path = '.'
        return ['-I=' + path]

    def compute_parameters_with_absolute_paths(self, parameter_list: T.List[str], build_dir: str) -> T.List[str]:
        for idx, i in enumerate(parameter_list):
            if i[:3] == '-I=':
                parameter_list[idx] = i[:3] + os.path.normpath(os.path.join(build_dir, i[3:]))
            if i[:4] == '-L-L':
                parameter_list[idx] = i[:4] + os.path.normpath(os.path.join(build_dir, i[4:]))
            if i[:5] == '-L=-L':
                parameter_list[idx] = i[:5] + os.path.normpath(os.path.join(build_dir, i[5:]))
            if i[:6] == '-Wl,-L':
                parameter_list[idx] = i[:6] + os.path.normpath(os.path.join(build_dir, i[6:]))
        return parameter_list

    def get_warn_args(self, level: str) -> T.List[str]:
        return ['-wi']

    def get_werror_args(self) -> T.List[str]:
        return ['-w']

    def get_coverage_args(self) -> T.List[str]:
        return ['-cov']

    def get_coverage_link_args(self) -> T.List[str]:
        return []

    def get_preprocess_only_args(self) -> T.List[str]:
        return ['-E']

    def get_compile_only_args(self) -> T.List[str]:
        return ['-c']

    def get_depfile_suffix(self) -> str:
        return 'deps'

    def get_dependency_gen_args(self, outtarget: str, outfile: str) -> T.List[str]:
        if self._dmd_has_depfile:
            return [f'-makedeps={outfile}']
        return []

    def get_pic_args(self) -> T.List[str]:
        if self.info.is_windows():
            return []
        return ['-fPIC']

    def get_optimization_link_args(self, optimization_level: str) -> T.List[str]:
        if optimization_level != 'plain':
            return self._get_target_arch_args()
        return []

    def gen_import_library_args(self, implibname: str) -> T.List[str]:
        return self.linker.import_library_args(implibname)

    def build_rpath_args(self, env: 'Environment', build_dir: str, from_dir: str, rpath_paths: T.Tuple[str, ...], build_rpath: str, install_rpath: str) -> T.Tuple[T.List[str], T.Set[bytes]]:
        if self.info.is_windows():
            return ([], set())
        if self.linker.id.startswith('ld'):
            args: T.List[str] = []
            rpath_args, rpath_dirs_to_remove = super().build_rpath_args(env, build_dir, from_dir, rpath_paths, build_rpath, install_rpath)
            for r in rpath_args:
                if ',' in r:
                    a, b = r.split(',', maxsplit=1)
                    args.append(a)
                    args.append(self.LINKER_PREFIX + b)
                else:
                    args.append(r)
            return (args, rpath_dirs_to_remove)
        return super().build_rpath_args(env, build_dir, from_dir, rpath_paths, build_rpath, install_rpath)

    @classmethod
    def _translate_args_to_nongnu(cls, args: T.List[str], info: MachineInfo, link_id: str) -> T.List[str]:
        dcargs: T.List[str] = []
        link_expect_arg = False
        link_flags_with_arg = ['-rpath', '-rpath-link', '-soname', '-compatibility_version', '-current_version']
        for arg in args:
            osargs: T.List[str] = []
            if info.is_windows():
                osargs = cls.translate_arg_to_windows(arg)
            elif info.is_darwin():
                osargs = cls._translate_arg_to_osx(arg)
            if osargs:
                dcargs.extend(osargs)
                continue
            if arg == '-pthread':
                continue
            if arg.startswith('-fstack-protector'):
                continue
            if arg.startswith('-D') and (not (arg == '-D' or arg.startswith(('-Dd', '-Df')))):
                continue
            if arg.startswith('-Wl,'):
                linkargs = arg[arg.index(',') + 1:].split(',')
                for la in linkargs:
                    dcargs.append('-L=' + la.strip())
                continue
            elif arg.startswith(('-link-defaultlib', '-linker', '-link-internally', '-linkonce-templates', '-lib')):
                dcargs.append(arg)
                continue
            elif arg.startswith('-l'):
                dcargs.append('-L=' + arg)
                continue
            elif arg.startswith('-isystem'):
                if arg.startswith('-isystem='):
                    dcargs.append('-I=' + arg[9:])
                else:
                    dcargs.append('-I' + arg[8:])
                continue
            elif arg.startswith('-idirafter'):
                if arg.startswith('-idirafter='):
                    dcargs.append('-I=' + arg[11:])
                else:
                    dcargs.append('-I' + arg[10:])
                continue
            elif arg.startswith('-L'):
                if arg.startswith('-L='):
                    suffix = arg[3:]
                else:
                    suffix = arg[2:]
                if link_expect_arg:
                    dcargs.append(arg)
                    link_expect_arg = False
                    continue
                if suffix in link_flags_with_arg:
                    link_expect_arg = True
                if suffix.startswith('-') or suffix.startswith('@'):
                    dcargs.append(arg)
                    continue
                if info.is_windows() and link_id == 'link' and suffix.startswith('/'):
                    dcargs.append(arg)
                    continue
                if arg.endswith('.a') or arg.endswith('.lib'):
                    if len(suffix) > 0 and (not suffix.startswith('-')):
                        dcargs.append('-L=' + suffix)
                        continue
                dcargs.append('-L=' + arg)
                continue
            elif not arg.startswith('-') and arg.endswith(('.a', '.lib')):
                dcargs.append('-L=' + arg)
                continue
            else:
                dcargs.append(arg)
        return dcargs

    @classmethod
    def translate_arg_to_windows(cls, arg: str) -> T.List[str]:
        args: T.List[str] = []
        if arg.startswith('-Wl,'):
            linkargs = arg[arg.index(',') + 1:].split(',')
            for la in linkargs:
                if la.startswith('--out-implib='):
                    args.append('-L=/IMPLIB:' + la[13:].strip())
        elif arg.startswith('-mscrtlib='):
            args.append(arg)
            mscrtlib = arg[10:].lower()
            if cls is LLVMDCompiler:
                if mscrtlib != 'libcmt':
                    args.append('-L=/NODEFAULTLIB:libcmt')
                    args.append('-L=/NODEFAULTLIB:libvcruntime')
                if mscrtlib.startswith('msvcrt'):
                    args.append('-L=/DEFAULTLIB:legacy_stdio_definitions.lib')
        return args

    @classmethod
    def _translate_arg_to_osx(cls, arg: str) -> T.List[str]:
        args: T.List[str] = []
        if arg.startswith('-install_name'):
            args.append('-L=' + arg)
        return args

    @classmethod
    def _unix_args_to_native(cls, args: T.List[str], info: MachineInfo, link_id: str='') -> T.List[str]:
        return cls._translate_args_to_nongnu(args, info, link_id)

    def get_debug_args(self, is_debug: bool) -> T.List[str]:
        ddebug_args = []
        if is_debug:
            ddebug_args = [d_feature_args[self.id]['debug']]
        return clike_debug_args[is_debug] + ddebug_args

    def _get_crt_args(self, crt_val: str, buildtype: str) -> T.List[str]:
        if not self.info.is_windows():
            return []
        return self.mscrt_args[self.get_crt_val(crt_val, buildtype)]

    def get_soname_args(self, env: 'Environment', prefix: str, shlib_name: str, suffix: str, soversion: str, darwin_versions: T.Tuple[str, str]) -> T.List[str]:
        sargs = super().get_soname_args(env, prefix, shlib_name, suffix, soversion, darwin_versions)
        soargs: T.List[str] = []
        if self.linker.id.startswith('ld.'):
            for arg in sargs:
                a, b = arg.split(',', maxsplit=1)
                soargs.append(a)
                soargs.append(self.LINKER_PREFIX + b)
            return soargs
        elif self.linker.id.startswith('ld64'):
            for arg in sargs:
                if not arg.startswith(self.LINKER_PREFIX):
                    soargs.append(self.LINKER_PREFIX + arg)
                else:
                    soargs.append(arg)
            return soargs
        else:
            return sargs

    def get_allow_undefined_link_args(self) -> T.List[str]:
        args = self.linker.get_allow_undefined_args()
        if self.info.is_darwin():
            args = [a.replace('-L=', '-Xcc=-Wl,') for a in args]
        return args