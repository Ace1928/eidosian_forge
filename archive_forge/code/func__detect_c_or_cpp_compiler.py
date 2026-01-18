from __future__ import annotations
from ..mesonlib import (
from ..envconfig import BinaryTable
from .. import mlog
from ..linkers import guess_win_linker, guess_nix_linker
import subprocess
import platform
import re
import shutil
import tempfile
import os
import typing as T
def _detect_c_or_cpp_compiler(env: 'Environment', lang: str, for_machine: MachineChoice, *, override_compiler: T.Optional[T.List[str]]=None) -> Compiler:
    """Shared implementation for finding the C or C++ compiler to use.

    the override_compiler option is provided to allow compilers which use
    the compiler (GCC or Clang usually) as their shared linker, to find
    the linker they need.
    """
    from . import c, cpp
    from ..linkers import linkers
    popen_exceptions: T.Dict[str, T.Union[Exception, str]] = {}
    compilers, ccache, exe_wrap = _get_compilers(env, lang, for_machine)
    if override_compiler is not None:
        compilers = [override_compiler]
    is_cross = env.is_cross_build(for_machine)
    info = env.machines[for_machine]
    cls: T.Union[T.Type[CCompiler], T.Type[CPPCompiler]]
    lnk: T.Union[T.Type[StaticLinker], T.Type[DynamicLinker]]
    for compiler in compilers:
        if isinstance(compiler, str):
            compiler = [compiler]
        compiler_name = os.path.basename(compiler[0])
        if any((os.path.basename(x) in {'cl', 'cl.exe', 'clang-cl', 'clang-cl.exe'} for x in compiler)):
            if 'WATCOM' in os.environ:

                def sanitize(p: str) -> str:
                    return os.path.normcase(os.path.abspath(p))
                watcom_cls = [sanitize(os.path.join(os.environ['WATCOM'], 'BINNT', 'cl')), sanitize(os.path.join(os.environ['WATCOM'], 'BINNT', 'cl.exe')), sanitize(os.path.join(os.environ['WATCOM'], 'BINNT64', 'cl')), sanitize(os.path.join(os.environ['WATCOM'], 'BINNT64', 'cl.exe'))]
                found_cl = sanitize(shutil.which('cl'))
                if found_cl in watcom_cls:
                    mlog.debug('Skipping unsupported cl.exe clone at:', found_cl)
                    continue
            arg = '/?'
        elif 'armcc' in compiler_name:
            arg = '--vsn'
        elif 'ccrx' in compiler_name:
            arg = '-v'
        elif 'xc16' in compiler_name:
            arg = '--version'
        elif 'ccomp' in compiler_name:
            arg = '-version'
        elif compiler_name in {'cl2000', 'cl2000.exe', 'cl430', 'cl430.exe', 'armcl', 'armcl.exe'}:
            arg = '-version'
        elif compiler_name in {'icl', 'icl.exe'}:
            arg = ''
        else:
            arg = '--version'
        cmd = compiler + [arg]
        try:
            p, out, err = Popen_safe_logged(cmd, msg='Detecting compiler via')
        except OSError as e:
            popen_exceptions[join_args(cmd)] = e
            continue
        if 'ccrx' in compiler_name:
            out = err
        full_version = out.split('\n', 1)[0]
        version = search_version(out)
        guess_gcc_or_lcc: T.Optional[str] = None
        if 'Free Software Foundation' in out or out.startswith('xt-'):
            guess_gcc_or_lcc = 'gcc'
        if 'e2k' in out and 'lcc' in out:
            guess_gcc_or_lcc = 'lcc'
        if 'Microchip Technology' in out:
            guess_gcc_or_lcc = None
        if guess_gcc_or_lcc:
            defines = _get_gnu_compiler_defines(compiler)
            if not defines:
                popen_exceptions[join_args(compiler)] = 'no pre-processor defines'
                continue
            if guess_gcc_or_lcc == 'lcc':
                version = _get_lcc_version_from_defines(defines)
                cls = c.ElbrusCCompiler if lang == 'c' else cpp.ElbrusCPPCompiler
            else:
                version = _get_gnu_version_from_defines(defines)
                cls = c.GnuCCompiler if lang == 'c' else cpp.GnuCPPCompiler
            linker = guess_nix_linker(env, compiler, cls, version, for_machine)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, defines=defines, full_version=full_version, linker=linker)
        if 'Emscripten' in out:
            cls = c.EmscriptenCCompiler if lang == 'c' else cpp.EmscriptenCPPCompiler
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            with tempfile.NamedTemporaryFile(suffix='.c') as f:
                cmd = compiler + [cls.LINKER_PREFIX + '--version', f.name]
                _, o, _ = Popen_safe(cmd)
            linker = linkers.WASMDynamicLinker(compiler, for_machine, cls.LINKER_PREFIX, [], version=search_version(o))
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, linker=linker, full_version=full_version)
        if 'Arm C/C++/Fortran Compiler' in out:
            arm_ver_match = re.search('version (\\d+)\\.(\\d+)\\.?(\\d+)? \\(build number (\\d+)\\)', out)
            assert arm_ver_match is not None, 'for mypy'
            version = '.'.join([x for x in arm_ver_match.groups() if x is not None])
            if lang == 'c':
                cls = c.ArmLtdClangCCompiler
            elif lang == 'cpp':
                cls = cpp.ArmLtdClangCPPCompiler
            linker = guess_nix_linker(env, compiler, cls, version, for_machine)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, linker=linker)
        if 'armclang' in out:
            arm_ver_match = re.search('.*Component.*', out)
            if arm_ver_match is None:
                popen_exceptions[join_args(compiler)] = 'version string not found'
                continue
            arm_ver_str = arm_ver_match.group(0)
            version = search_version(arm_ver_str)
            full_version = arm_ver_str
            cls = c.ArmclangCCompiler if lang == 'c' else cpp.ArmclangCPPCompiler
            linker = linkers.ArmClangDynamicLinker(for_machine, version=version)
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
        if 'CL.EXE COMPATIBILITY' in out:
            arg = '--version'
            try:
                p, out, err = Popen_safe(compiler + [arg])
            except OSError as e:
                popen_exceptions[join_args(compiler + [arg])] = e
            version = search_version(out)
            match = re.search('^Target: (.*?)-', out, re.MULTILINE)
            if match:
                target = match.group(1)
            else:
                target = 'unknown target'
            cls = c.ClangClCCompiler if lang == 'c' else cpp.ClangClCPPCompiler
            linker = guess_win_linker(env, ['lld-link'], cls, version, for_machine)
            return cls(compiler, version, for_machine, is_cross, info, target, exe_wrap, linker=linker)
        if 'clang' in out or 'Clang' in out:
            linker = None
            defines = _get_clang_compiler_defines(compiler)
            if 'Apple' in out:
                cls = c.AppleClangCCompiler if lang == 'c' else cpp.AppleClangCPPCompiler
            else:
                cls = c.ClangCCompiler if lang == 'c' else cpp.ClangCPPCompiler
            if 'windows' in out or env.machines[for_machine].is_windows():
                try:
                    linker = guess_win_linker(env, compiler, cls, version, for_machine, invoked_directly=False)
                except MesonException:
                    pass
            if linker is None:
                linker = guess_nix_linker(env, compiler, cls, version, for_machine)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, defines=defines, full_version=full_version, linker=linker)
        if 'Intel(R) C++ Intel(R)' in err:
            version = search_version(err)
            target = 'x86' if 'IA-32' in err else 'x86_64'
            cls = c.IntelClCCompiler if lang == 'c' else cpp.IntelClCPPCompiler
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            linker = linkers.XilinkDynamicLinker(for_machine, [], version=version)
            return cls(compiler, version, for_machine, is_cross, info, target, exe_wrap, linker=linker)
        if 'Intel(R) oneAPI DPC++/C++ Compiler for applications' in err:
            version = search_version(err)
            target = 'x86' if 'IA-32' in err else 'x86_64'
            cls = c.IntelLLVMClCCompiler if lang == 'c' else cpp.IntelLLVMClCPPCompiler
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            linker = linkers.XilinkDynamicLinker(for_machine, [], version=version)
            return cls(compiler, version, for_machine, is_cross, info, target, exe_wrap, linker=linker)
        if 'Microsoft' in out or 'Microsoft' in err:
            for lookat in [err, out]:
                version = search_version(lookat)
                if version != 'unknown version':
                    break
            else:
                raise EnvironmentException(f'Failed to detect MSVC compiler version: stderr was\n{err!r}')
            cl_signature = lookat.split('\n', maxsplit=1)[0]
            match = re.search('.*(x86|x64|ARM|ARM64)([^_A-Za-z0-9]|$)', cl_signature)
            if match:
                target = match.group(1)
            else:
                m = f"Failed to detect MSVC compiler target architecture: 'cl /?' output is\n{cl_signature}"
                raise EnvironmentException(m)
            cls = c.VisualStudioCCompiler if lang == 'c' else cpp.VisualStudioCPPCompiler
            linker = guess_win_linker(env, ['link'], cls, version, for_machine)
            if 'sccache' not in ccache:
                ccache = []
            return cls(ccache, compiler, version, for_machine, is_cross, info, target, exe_wrap, full_version=cl_signature, linker=linker)
        if 'PGI Compilers' in out:
            cls = c.PGICCompiler if lang == 'c' else cpp.PGICPPCompiler
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            linker = linkers.PGIDynamicLinker(compiler, for_machine, cls.LINKER_PREFIX, [], version=version)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, linker=linker)
        if 'NVIDIA Compilers and Tools' in out:
            cls = c.NvidiaHPC_CCompiler if lang == 'c' else cpp.NvidiaHPC_CPPCompiler
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            linker = linkers.NvidiaHPC_DynamicLinker(compiler, for_machine, cls.LINKER_PREFIX, [], version=version)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, linker=linker)
        if '(ICC)' in out:
            cls = c.IntelCCompiler if lang == 'c' else cpp.IntelCPPCompiler
            l = guess_nix_linker(env, compiler, cls, version, for_machine)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=l)
        if 'Intel(R) oneAPI' in out:
            cls = c.IntelLLVMCCompiler if lang == 'c' else cpp.IntelLLVMCPPCompiler
            l = guess_nix_linker(env, compiler, cls, version, for_machine)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=l)
        if 'TMS320C2000 C/C++' in out or 'MSP430 C/C++' in out or 'TI ARM C/C++ Compiler' in out:
            if 'TMS320C2000 C/C++' in out:
                cls = c.C2000CCompiler if lang == 'c' else cpp.C2000CPPCompiler
                lnk = linkers.C2000DynamicLinker
            else:
                cls = c.TICCompiler if lang == 'c' else cpp.TICPPCompiler
                lnk = linkers.TIDynamicLinker
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            linker = lnk(compiler, for_machine, version=version)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
        if 'ARM' in out and (not ('Metrowerks' in out or 'Freescale' in out)):
            cls = c.ArmCCompiler if lang == 'c' else cpp.ArmCPPCompiler
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            linker = linkers.ArmDynamicLinker(for_machine, version=version)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
        if 'RX Family' in out:
            cls = c.CcrxCCompiler if lang == 'c' else cpp.CcrxCPPCompiler
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            linker = linkers.CcrxDynamicLinker(for_machine, version=version)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
        if 'Microchip Technology' in out:
            cls = c.Xc16CCompiler
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            linker = linkers.Xc16DynamicLinker(for_machine, version=version)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
        if 'CompCert' in out:
            cls = c.CompCertCCompiler
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            linker = linkers.CompCertDynamicLinker(for_machine, version=version)
            return cls(ccache, compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
        if 'Metrowerks C/C++' in out or 'Freescale C/C++' in out:
            if 'ARM' in out:
                cls = c.MetrowerksCCompilerARM if lang == 'c' else cpp.MetrowerksCPPCompilerARM
                lnk = linkers.MetrowerksLinkerARM
            else:
                cls = c.MetrowerksCCompilerEmbeddedPowerPC if lang == 'c' else cpp.MetrowerksCPPCompilerEmbeddedPowerPC
                lnk = linkers.MetrowerksLinkerEmbeddedPowerPC
            mwcc_ver_match = re.search('Version (\\d+)\\.(\\d+)\\.?(\\d+)? build (\\d+)', out)
            assert mwcc_ver_match is not None, 'for mypy'
            compiler_version = '.'.join((x for x in mwcc_ver_match.groups() if x is not None))
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            ld = env.lookup_binary_entry(for_machine, cls.language + '_ld')
            if ld is not None:
                _, o_ld, _ = Popen_safe(ld + ['--version'])
                mwld_ver_match = re.search('Version (\\d+)\\.(\\d+)\\.?(\\d+)? build (\\d+)', o_ld)
                assert mwld_ver_match is not None, 'for mypy'
                linker_version = '.'.join((x for x in mwld_ver_match.groups() if x is not None))
                linker = lnk(ld, for_machine, version=linker_version)
            else:
                raise EnvironmentException(f'Failed to detect linker for {cls.id!r} compiler. Please update your cross file(s).')
            return cls(ccache, compiler, compiler_version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
    _handle_exceptions(popen_exceptions, compilers)
    raise EnvironmentException(f'Unknown compiler {compilers}')