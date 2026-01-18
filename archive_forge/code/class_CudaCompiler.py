from __future__ import annotations
import enum
import os.path
import string
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import (
from .compilers import Compiler
class CudaCompiler(Compiler):
    LINKER_PREFIX = '-Xlinker='
    language = 'cuda'
    _FLAG_PASSTHRU_NOARGS = {'--objdir-as-tempdir', '-objtemp', '--generate-dependency-targets', '-MP', '--allow-unsupported-compiler', '-allow-unsupported-compiler', '--link', '--lib', '-lib', '--device-link', '-dlink', '--device-c', '-dc', '--device-w', '-dw', '--cuda', '-cuda', '--compile', '-c', '--fatbin', '-fatbin', '--cubin', '-cubin', '--ptx', '-ptx', '--preprocess', '-E', '--generate-dependencies', '-M', '--generate-nonsystem-dependencies', '-MM', '--generate-dependencies-with-compile', '-MD', '--generate-nonsystem-dependencies-with-compile', '-MMD', '--run', '--profile', '-pg', '--debug', '-g', '--device-debug', '-G', '--extensible-whole-program', '-ewp', '--generate-line-info', '-lineinfo', '--dlink-time-opt', '-dlto', '--no-exceptions', '-noeh', '--shared', '-shared', '--no-host-device-initializer-list', '-nohdinitlist', '--expt-relaxed-constexpr', '-expt-relaxed-constexpr', '--extended-lambda', '-extended-lambda', '--expt-extended-lambda', '-expt-extended-lambda', '--m32', '-m32', '--m64', '-m64', '--forward-unknown-to-host-compiler', '-forward-unknown-to-host-compiler', '--forward-unknown-to-host-linker', '-forward-unknown-to-host-linker', '--dont-use-profile', '-noprof', '--dryrun', '-dryrun', '--verbose', '-v', '--keep', '-keep', '--save-temps', '-save-temps', '--clean-targets', '-clean', '--no-align-double', '--no-device-link', '-nodlink', '--allow-unsupported-compiler', '-allow-unsupported-compiler', '--use_fast_math', '-use_fast_math', '--extra-device-vectorization', '-extra-device-vectorization', '--compile-as-tools-patch', '-astoolspatch', '--keep-device-functions', '-keep-device-functions', '--disable-warnings', '-w', '--source-in-ptx', '-src-in-ptx', '--restrict', '-restrict', '--Wno-deprecated-gpu-targets', '-Wno-deprecated-gpu-targets', '--Wno-deprecated-declarations', '-Wno-deprecated-declarations', '--Wreorder', '-Wreorder', '--Wdefault-stream-launch', '-Wdefault-stream-launch', '--Wext-lambda-captures-this', '-Wext-lambda-captures-this', '--display-error-number', '-err-no', '--resource-usage', '-res-usage', '--help', '-h', '--version', '-V', '--list-gpu-code', '-code-ls', '--list-gpu-arch', '-arch-ls'}
    _FLAG_LONG2SHORT_WITHARGS = {'--output-file': '-o', '--pre-include': '-include', '--library': '-l', '--define-macro': '-D', '--undefine-macro': '-U', '--include-path': '-I', '--system-include': '-isystem', '--library-path': '-L', '--output-directory': '-odir', '--dependency-output': '-MF', '--compiler-bindir': '-ccbin', '--archiver-binary': '-arbin', '--cudart': '-cudart', '--cudadevrt': '-cudadevrt', '--libdevice-directory': '-ldir', '--target-directory': '-target-dir', '--optimization-info': '-opt-info', '--optimize': '-O', '--ftemplate-backtrace-limit': '-ftemplate-backtrace-limit', '--ftemplate-depth': '-ftemplate-depth', '--x': '-x', '--std': '-std', '--machine': '-m', '--compiler-options': '-Xcompiler', '--linker-options': '-Xlinker', '--archive-options': '-Xarchive', '--ptxas-options': '-Xptxas', '--nvlink-options': '-Xnvlink', '--threads': '-t', '--keep-dir': '-keep-dir', '--run-args': '-run-args', '--input-drive-prefix': '-idp', '--dependency-drive-prefix': '-ddp', '--drive-prefix': '-dp', '--dependency-target-name': '-MT', '--default-stream': '-default-stream', '--gpu-architecture': '-arch', '--gpu-code': '-code', '--generate-code': '-gencode', '--relocatable-device-code': '-rdc', '--entries': '-e', '--maxrregcount': '-maxrregcount', '--ftz': '-ftz', '--prec-div': '-prec-div', '--prec-sqrt': '-prec-sqrt', '--fmad': '-fmad', '--Werror': '-Werror', '--diag-error': '-diag-error', '--diag-suppress': '-diag-suppress', '--diag-warn': '-diag-warn', '--options-file': '-optf', '--time': '-time', '--qpp-config': '-qpp-config'}
    _FLAG_SHORT2LONG_WITHARGS = {v: k for k, v in _FLAG_LONG2SHORT_WITHARGS.items()}
    id = 'nvcc'

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, exe_wrapper: T.Optional['ExternalProgram'], host_compiler: Compiler, info: 'MachineInfo', linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None):
        super().__init__(ccache, exelist, version, for_machine, info, linker=linker, full_version=full_version, is_cross=is_cross)
        self.exe_wrapper = exe_wrapper
        self.host_compiler = host_compiler
        self.base_options = host_compiler.base_options
        self.warn_args = {level: self._to_host_flags(list((f for f in flags if f != '-Wpedantic'))) for level, flags in host_compiler.warn_args.items()}
        self.host_werror_args = ['-Xcompiler=' + x for x in self.host_compiler.get_werror_args()]

    @classmethod
    def _shield_nvcc_list_arg(cls, arg: str, listmode: bool=True) -> str:
        """
        Shield an argument against both splitting by NVCC's list-argument
        parse logic, and interpretation by any shell.

        NVCC seems to consider every comma , that is neither escaped by \\ nor inside
        a double-quoted string a split-point. Single-quotes do not provide protection
        against splitting; In fact, after splitting they are \\-escaped. Unfortunately,
        double-quotes don't protect against shell expansion. What follows is a
        complex dance to accommodate everybody.
        """
        SQ = "'"
        DQ = '"'
        CM = ','
        BS = '\\'
        DQSQ = DQ + SQ + DQ
        quotable = set(string.whitespace + '"$`\\')
        if CM not in arg or not listmode:
            if SQ not in arg:
                if set(arg).intersection(quotable):
                    return SQ + arg + SQ
                else:
                    return arg
            else:
                l = [cls._shield_nvcc_list_arg(s) for s in arg.split(SQ)]
                l = sum([[s, DQSQ] for s in l][:-1], [])
                return ''.join(l)
        else:
            l = ['']
            instring = False
            argit = iter(arg)
            for c in argit:
                if c == CM and (not instring):
                    l.append('')
                elif c == DQ:
                    l[-1] += c
                    instring = not instring
                elif c == BS:
                    try:
                        l[-1] += next(argit)
                    except StopIteration:
                        break
                else:
                    l[-1] += c
            l = [cls._shield_nvcc_list_arg(s, listmode=False) for s in l]
            return '\\,'.join(l)

    @classmethod
    def _merge_flags(cls, flags: T.List[str]) -> T.List[str]:
        """
        The flags to NVCC gets exceedingly verbose and unreadable when too many of them
        are shielded with -Xcompiler. Merge consecutive -Xcompiler-wrapped arguments
        into one.
        """
        if len(flags) <= 1:
            return flags
        flagit = iter(flags)
        xflags = []

        def is_xcompiler_flag_isolated(flag: str) -> bool:
            return flag == '-Xcompiler'

        def is_xcompiler_flag_glued(flag: str) -> bool:
            return flag.startswith('-Xcompiler=')

        def is_xcompiler_flag(flag: str) -> bool:
            return is_xcompiler_flag_isolated(flag) or is_xcompiler_flag_glued(flag)

        def get_xcompiler_val(flag: str, flagit: T.Iterator[str]) -> str:
            if is_xcompiler_flag_glued(flag):
                return flag[len('-Xcompiler='):]
            else:
                try:
                    return next(flagit)
                except StopIteration:
                    return ''
        ingroup = False
        for flag in flagit:
            if not is_xcompiler_flag(flag):
                ingroup = False
                xflags.append(flag)
            elif ingroup:
                xflags[-1] += ','
                xflags[-1] += get_xcompiler_val(flag, flagit)
            elif is_xcompiler_flag_isolated(flag):
                ingroup = True
                xflags.append(flag)
                xflags.append(get_xcompiler_val(flag, flagit))
            elif is_xcompiler_flag_glued(flag):
                ingroup = True
                xflags.append(flag)
            else:
                raise ValueError('-Xcompiler flag merging failed, unknown argument form!')
        return xflags

    def _to_host_flags(self, flags: T.List[str], phase: _Phase=_Phase.COMPILER) -> T.List[str]:
        """
        Translate generic "GCC-speak" plus particular "NVCC-speak" flags to NVCC flags.

        NVCC's "short" flags have broad similarities to the GCC standard, but have
        gratuitous, irritating differences.
        """
        xflags = []
        flagit = iter(flags)
        for flag in flagit:
            if flag in self._FLAG_PASSTHRU_NOARGS:
                xflags.append(flag)
                continue
            if flag[:1] not in '-/':
                xflags.append(flag)
                continue
            elif flag[:1] == '/':
                wrap = '"' if ',' in flag else ''
                xflags.append(f'-X{phase.value}={wrap}{flag}{wrap}')
                continue
            elif len(flag) >= 2 and flag[0] == '-' and (flag[1] in 'IDULlmOxmte'):
                if flag[2:3] == '':
                    try:
                        val = next(flagit)
                    except StopIteration:
                        pass
                elif flag[2:3] == '=':
                    val = flag[3:]
                else:
                    val = flag[2:]
                flag = flag[:2]
            elif flag in self._FLAG_LONG2SHORT_WITHARGS or flag in self._FLAG_SHORT2LONG_WITHARGS:
                try:
                    val = next(flagit)
                except StopIteration:
                    pass
            elif flag.split('=', 1)[0] in self._FLAG_LONG2SHORT_WITHARGS or flag.split('=', 1)[0] in self._FLAG_SHORT2LONG_WITHARGS:
                flag, val = flag.split('=', 1)
            elif flag.startswith('-isystem'):
                val = flag[8:].strip()
                flag = flag[:8]
            else:
                if flag == '-ffast-math':
                    xflags.append('-use_fast_math')
                    xflags.append('-Xcompiler=' + flag)
                elif flag == '-fno-fast-math':
                    xflags.append('-ftz=false')
                    xflags.append('-prec-div=true')
                    xflags.append('-prec-sqrt=true')
                    xflags.append('-Xcompiler=' + flag)
                elif flag == '-freciprocal-math':
                    xflags.append('-prec-div=false')
                    xflags.append('-Xcompiler=' + flag)
                elif flag == '-fno-reciprocal-math':
                    xflags.append('-prec-div=true')
                    xflags.append('-Xcompiler=' + flag)
                else:
                    xflags.append('-Xcompiler=' + self._shield_nvcc_list_arg(flag))
                continue
            assert val is not None
            flag = self._FLAG_LONG2SHORT_WITHARGS.get(flag, flag)
            if flag in {'-include', '-isystem', '-I', '-L', '-l'}:
                if len(flag) == 2:
                    xflags.append(flag + self._shield_nvcc_list_arg(val))
                elif flag == '-isystem' and val in self.host_compiler.get_default_include_dirs():
                    pass
                else:
                    xflags.append(flag)
                    xflags.append(self._shield_nvcc_list_arg(val))
            elif flag == '-O':
                if val == 'fast':
                    xflags.append('-O3')
                    xflags.append('-use_fast_math')
                    xflags.append('-Xcompiler')
                    xflags.append(flag + val)
                elif val in {'s', 'g', 'z'}:
                    xflags.append('-Xcompiler')
                    xflags.append(flag + val)
                else:
                    xflags.append(flag + val)
            elif flag in {'-D', '-U', '-m', '-t'}:
                xflags.append(flag + val)
            elif flag in {'-std'}:
                xflags.append(flag + '=' + val)
            else:
                xflags.append(flag)
                xflags.append(val)
        return self._merge_flags(xflags)

    def needs_static_linker(self) -> bool:
        return False

    def thread_link_flags(self, environment: 'Environment') -> T.List[str]:
        return self._to_host_flags(self.host_compiler.thread_link_flags(environment), _Phase.LINKER)

    def sanity_check(self, work_dir: str, env: 'Environment') -> None:
        mlog.debug('Sanity testing ' + self.get_display_language() + ' compiler:', ' '.join(self.exelist))
        mlog.debug('Is cross compiler: %s.' % str(self.is_cross))
        sname = 'sanitycheckcuda.cu'
        code = '\n        #include <cuda_runtime.h>\n        #include <stdio.h>\n\n        __global__ void kernel (void) {}\n\n        int main(void){\n            struct cudaDeviceProp prop;\n            int count, i;\n            cudaError_t ret = cudaGetDeviceCount(&count);\n            if(ret != cudaSuccess){\n                fprintf(stderr, "%d\\n", (int)ret);\n            }else{\n                for(i=0;i<count;i++){\n                    if(cudaGetDeviceProperties(&prop, i) == cudaSuccess){\n                        fprintf(stdout, "%d.%d\\n", prop.major, prop.minor);\n                    }\n                }\n            }\n            fflush(stderr);\n            fflush(stdout);\n            return 0;\n        }\n        '
        binname = sname.rsplit('.', 1)[0]
        binname += '_cross' if self.is_cross else ''
        source_name = os.path.join(work_dir, sname)
        binary_name = os.path.join(work_dir, binname + '.exe')
        with open(source_name, 'w', encoding='utf-8') as ofile:
            ofile.write(code)
        self.detected_cc = ''
        flags = []
        flags += ['-w', '-cudart', 'static', source_name]
        flags += self.get_ccbin_args(env.coredata.options)
        if self.is_cross and self.exe_wrapper is None:
            flags += self.get_compile_only_args()
        flags += self.get_output_args(binary_name)
        cmdlist = self.exelist + flags
        mlog.debug('Sanity check compiler command line: ', ' '.join(cmdlist))
        pc, stdo, stde = Popen_safe(cmdlist, cwd=work_dir)
        mlog.debug('Sanity check compile stdout: ')
        mlog.debug(stdo)
        mlog.debug('-----\nSanity check compile stderr:')
        mlog.debug(stde)
        mlog.debug('-----')
        if pc.returncode != 0:
            raise EnvironmentException(f'Compiler {self.name_string()} cannot compile programs.')
        if self.is_cross:
            if self.exe_wrapper is None:
                return
            else:
                cmdlist = self.exe_wrapper.get_command() + [binary_name]
        else:
            cmdlist = self.exelist + ['--run', '"' + binary_name + '"']
        mlog.debug('Sanity check run command line: ', ' '.join(cmdlist))
        pe, stdo, stde = Popen_safe(cmdlist, cwd=work_dir)
        mlog.debug('Sanity check run stdout: ')
        mlog.debug(stdo)
        mlog.debug('-----\nSanity check run stderr:')
        mlog.debug(stde)
        mlog.debug('-----')
        pe.wait()
        if pe.returncode != 0:
            raise EnvironmentException(f'Executables created by {self.language} compiler {self.name_string()} are not runnable.')
        if stde == '':
            self.detected_cc = stdo
        else:
            mlog.debug('cudaGetDeviceCount() returned ' + stde)

    def has_header_symbol(self, hname: str, symbol: str, prefix: str, env: 'Environment', *, extra_args: T.Union[None, T.List[str], T.Callable[[CompileCheckMode], T.List[str]]]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> T.Tuple[bool, bool]:
        if extra_args is None:
            extra_args = []
        fargs = {'prefix': prefix, 'header': hname, 'symbol': symbol}
        t = "{prefix}\n        #include <{header}>\n        int main(void) {{\n            /* If it's not defined as a macro, try to use as a symbol */\n            #ifndef {symbol}\n                {symbol};\n            #endif\n            return 0;\n        }}"
        found, cached = self.compiles(t.format_map(fargs), env, extra_args=extra_args, dependencies=dependencies)
        if found:
            return (True, cached)
        t = '{prefix}\n        #include <{header}>\n        using {symbol};\n        int main(void) {{\n            return 0;\n        }}'
        return self.compiles(t.format_map(fargs), env, extra_args=extra_args, dependencies=dependencies)
    _CPP14_VERSION = '>=9.0'
    _CPP17_VERSION = '>=11.0'
    _CPP20_VERSION = '>=12.0'

    def get_options(self) -> 'MutableKeyedOptionDictType':
        opts = super().get_options()
        std_key = OptionKey('std', machine=self.for_machine, lang=self.language)
        ccbindir_key = OptionKey('ccbindir', machine=self.for_machine, lang=self.language)
        cpp_stds = ['none', 'c++03', 'c++11']
        if version_compare(self.version, self._CPP14_VERSION):
            cpp_stds += ['c++14']
        if version_compare(self.version, self._CPP17_VERSION):
            cpp_stds += ['c++17']
        if version_compare(self.version, self._CPP20_VERSION):
            cpp_stds += ['c++20']
        opts.update({std_key: coredata.UserComboOption('C++ language standard to use with CUDA', cpp_stds, 'none'), ccbindir_key: coredata.UserStringOption('CUDA non-default toolchain directory to use (-ccbin)', '')})
        return opts

    def _to_host_compiler_options(self, options: 'KeyedOptionDictType') -> 'KeyedOptionDictType':
        """
        Convert an NVCC Option set to a host compiler's option set.
        """
        host_options = {key: options.get(key, opt) for key, opt in self.host_compiler.get_options().items()}
        std_key = OptionKey('std', machine=self.for_machine, lang=self.host_compiler.language)
        overrides = {std_key: 'none'}
        return coredata.OptionsView(host_options, overrides=overrides)

    def get_option_compile_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        args = self.get_ccbin_args(options)
        if not is_windows():
            key = OptionKey('std', machine=self.for_machine, lang=self.language)
            std = options[key]
            if std.value != 'none':
                args.append('--std=' + std.value)
        return args + self._to_host_flags(self.host_compiler.get_option_compile_args(self._to_host_compiler_options(options)))

    def get_option_link_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        args = self.get_ccbin_args(options)
        return args + self._to_host_flags(self.host_compiler.get_option_link_args(self._to_host_compiler_options(options)), _Phase.LINKER)

    def get_soname_args(self, env: 'Environment', prefix: str, shlib_name: str, suffix: str, soversion: str, darwin_versions: T.Tuple[str, str]) -> T.List[str]:
        return self._to_host_flags(self.host_compiler.get_soname_args(env, prefix, shlib_name, suffix, soversion, darwin_versions), _Phase.LINKER)

    def get_compile_only_args(self) -> T.List[str]:
        return ['-c']

    def get_no_optimization_args(self) -> T.List[str]:
        return ['-O0']

    def get_optimization_args(self, optimization_level: str) -> T.List[str]:
        return cuda_optimization_args[optimization_level]

    def sanitizer_compile_args(self, value: str) -> T.List[str]:
        return self._to_host_flags(self.host_compiler.sanitizer_compile_args(value))

    def sanitizer_link_args(self, value: str) -> T.List[str]:
        return self._to_host_flags(self.host_compiler.sanitizer_link_args(value))

    def get_debug_args(self, is_debug: bool) -> T.List[str]:
        return cuda_debug_args[is_debug]

    def get_werror_args(self) -> T.List[str]:
        device_werror_args = ['-Werror=cross-execution-space-call,deprecated-declarations,reorder']
        return device_werror_args + self.host_werror_args

    def get_warn_args(self, level: str) -> T.List[str]:
        return self.warn_args[level]

    def get_include_args(self, path: str, is_system: bool) -> T.List[str]:
        if path == '':
            path = '.'
        return ['-isystem=' + path] if is_system else ['-I' + path]

    def get_compile_debugfile_args(self, rel_obj: str, pch: bool=False) -> T.List[str]:
        return self._to_host_flags(self.host_compiler.get_compile_debugfile_args(rel_obj, pch))

    def get_link_debugfile_args(self, targetfile: str) -> T.List[str]:
        return self._to_host_flags(self.host_compiler.get_link_debugfile_args(targetfile), _Phase.LINKER)

    def get_depfile_suffix(self) -> str:
        return 'd'

    def get_optimization_link_args(self, optimization_level: str) -> T.List[str]:
        return self._to_host_flags(self.host_compiler.get_optimization_link_args(optimization_level), _Phase.LINKER)

    def build_rpath_args(self, env: 'Environment', build_dir: str, from_dir: str, rpath_paths: T.Tuple[str, ...], build_rpath: str, install_rpath: str) -> T.Tuple[T.List[str], T.Set[bytes]]:
        rpath_args, rpath_dirs_to_remove = self.host_compiler.build_rpath_args(env, build_dir, from_dir, rpath_paths, build_rpath, install_rpath)
        return (self._to_host_flags(rpath_args, _Phase.LINKER), rpath_dirs_to_remove)

    def linker_to_compiler_args(self, args: T.List[str]) -> T.List[str]:
        return args

    def get_pic_args(self) -> T.List[str]:
        return self._to_host_flags(self.host_compiler.get_pic_args())

    def compute_parameters_with_absolute_paths(self, parameter_list: T.List[str], build_dir: str) -> T.List[str]:
        return []

    def get_output_args(self, target: str) -> T.List[str]:
        return ['-o', target]

    def get_dependency_gen_args(self, outtarget: str, outfile: str) -> T.List[str]:
        if version_compare(self.version, '>= 10.2'):
            return ['-MD', '-MT', outtarget, '-MF', outfile]
        else:
            return []

    def get_std_exe_link_args(self) -> T.List[str]:
        return self._to_host_flags(self.host_compiler.get_std_exe_link_args(), _Phase.LINKER)

    def find_library(self, libname: str, env: 'Environment', extra_dirs: T.List[str], libtype: LibType=LibType.PREFER_SHARED, lib_prefix_warning: bool=True) -> T.Optional[T.List[str]]:
        return ['-l' + libname]

    def get_crt_compile_args(self, crt_val: str, buildtype: str) -> T.List[str]:
        return self._to_host_flags(self.host_compiler.get_crt_compile_args(crt_val, buildtype))

    def get_crt_link_args(self, crt_val: str, buildtype: str) -> T.List[str]:
        host_link_arg_overrides = []
        host_crt_compile_args = self.host_compiler.get_crt_compile_args(crt_val, buildtype)
        if any((arg in {'/MDd', '/MD', '/MTd'} for arg in host_crt_compile_args)):
            host_link_arg_overrides += ['/NODEFAULTLIB:LIBCMT.lib']
        return self._to_host_flags(host_link_arg_overrides + self.host_compiler.get_crt_link_args(crt_val, buildtype), _Phase.LINKER)

    def get_target_link_args(self, target: 'BuildTarget') -> T.List[str]:
        return self._to_host_flags(super().get_target_link_args(target), _Phase.LINKER)

    def get_dependency_compile_args(self, dep: 'Dependency') -> T.List[str]:
        return self._to_host_flags(super().get_dependency_compile_args(dep))

    def get_dependency_link_args(self, dep: 'Dependency') -> T.List[str]:
        return self._to_host_flags(super().get_dependency_link_args(dep), _Phase.LINKER)

    def get_ccbin_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        key = OptionKey('ccbindir', machine=self.for_machine, lang=self.language)
        ccbindir = options[key].value
        if isinstance(ccbindir, str) and ccbindir != '':
            return [self._shield_nvcc_list_arg('-ccbin=' + ccbindir, False)]
        else:
            return []

    def get_profile_generate_args(self) -> T.List[str]:
        return ['-Xcompiler=' + x for x in self.host_compiler.get_profile_generate_args()]

    def get_profile_use_args(self) -> T.List[str]:
        return ['-Xcompiler=' + x for x in self.host_compiler.get_profile_use_args()]

    def get_assert_args(self, disable: bool) -> T.List[str]:
        return self.host_compiler.get_assert_args(disable)