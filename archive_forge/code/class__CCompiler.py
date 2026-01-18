import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
class _CCompiler:
    """A helper class for `CCompilerOpt` containing all utilities that
    related to the fundamental compiler's functions.

    Attributes
    ----------
    cc_on_x86 : bool
        True when the target architecture is 32-bit x86
    cc_on_x64 : bool
        True when the target architecture is 64-bit x86
    cc_on_ppc64 : bool
        True when the target architecture is 64-bit big-endian powerpc
    cc_on_ppc64le : bool
        True when the target architecture is 64-bit litle-endian powerpc
    cc_on_s390x : bool
        True when the target architecture is IBM/ZARCH on linux
    cc_on_armhf : bool
        True when the target architecture is 32-bit ARMv7+
    cc_on_aarch64 : bool
        True when the target architecture is 64-bit Armv8-a+
    cc_on_noarch : bool
        True when the target architecture is unknown or not supported
    cc_is_gcc : bool
        True if the compiler is GNU or
        if the compiler is unknown
    cc_is_clang : bool
        True if the compiler is Clang
    cc_is_icc : bool
        True if the compiler is Intel compiler (unix like)
    cc_is_iccw : bool
        True if the compiler is Intel compiler (msvc like)
    cc_is_nocc : bool
        True if the compiler isn't supported directly,
        Note: that cause a fail-back to gcc
    cc_has_debug : bool
        True if the compiler has debug flags
    cc_has_native : bool
        True if the compiler has native flags
    cc_noopt : bool
        True if the compiler has definition 'DISABLE_OPT*',
        or 'cc_on_noarch' is True
    cc_march : str
        The target architecture name, or "unknown" if
        the architecture isn't supported
    cc_name : str
        The compiler name, or "unknown" if the compiler isn't supported
    cc_flags : dict
        Dictionary containing the initialized flags of `_Config.conf_cc_flags`
    """

    def __init__(self):
        if hasattr(self, 'cc_is_cached'):
            return
        detect_arch = (('cc_on_x64', '.*(x|x86_|amd)64.*', ''), ('cc_on_x86', '.*(win32|x86|i386|i686).*', ''), ('cc_on_ppc64le', '.*(powerpc|ppc)64(el|le).*|.*powerpc.*', 'defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)'), ('cc_on_ppc64', '.*(powerpc|ppc).*|.*powerpc.*', 'defined(__powerpc64__) && defined(__BIG_ENDIAN__)'), ('cc_on_aarch64', '.*(aarch64|arm64).*', ''), ('cc_on_armhf', '.*arm.*', 'defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__)'), ('cc_on_s390x', '.*s390x.*', ''), ('cc_on_noarch', '', ''))
        detect_compiler = (('cc_is_gcc', '.*(gcc|gnu\\-g).*', ''), ('cc_is_clang', '.*clang.*', ''), ('cc_is_iccw', '.*(intelw|intelemw|iccw).*', ''), ('cc_is_icc', '.*(intel|icc).*', ''), ('cc_is_msvc', '.*msvc.*', ''), ('cc_is_fcc', '.*fcc.*', ''), ('cc_is_nocc', '', ''))
        detect_args = (('cc_has_debug', '.*(O0|Od|ggdb|coverage|debug:full).*', ''), ('cc_has_native', '.*(-march=native|-xHost|/QxHost|-mcpu=a64fx).*', ''), ('cc_noopt', '.*DISABLE_OPT.*', ''))
        dist_info = self.dist_info()
        platform, compiler_info, extra_args = dist_info
        for section in (detect_arch, detect_compiler, detect_args):
            for attr, rgex, cexpr in section:
                setattr(self, attr, False)
        for detect, searchin in ((detect_arch, platform), (detect_compiler, compiler_info)):
            for attr, rgex, cexpr in detect:
                if rgex and (not re.match(rgex, searchin, re.IGNORECASE)):
                    continue
                if cexpr and (not self.cc_test_cexpr(cexpr)):
                    continue
                setattr(self, attr, True)
                break
        for attr, rgex, cexpr in detect_args:
            if rgex and (not re.match(rgex, extra_args, re.IGNORECASE)):
                continue
            if cexpr and (not self.cc_test_cexpr(cexpr)):
                continue
            setattr(self, attr, True)
        if self.cc_on_noarch:
            self.dist_log(f'unable to detect CPU architecture which lead to disable the optimization. check dist_info:<<\n{dist_info}\n>>', stderr=True)
            self.cc_noopt = True
        if self.conf_noopt:
            self.dist_log('Optimization is disabled by the Config', stderr=True)
            self.cc_noopt = True
        if self.cc_is_nocc:
            '\n            mingw can be treated as a gcc, and also xlc even if it based on clang,\n            but still has the same gcc optimization flags.\n            '
            self.dist_log(f"unable to detect compiler type which leads to treating it as GCC. this is a normal behavior if you're using gcc-like compiler such as MinGW or IBM/XLC.check dist_info:<<\n{dist_info}\n>>", stderr=True)
            self.cc_is_gcc = True
        self.cc_march = 'unknown'
        for arch in ('x86', 'x64', 'ppc64', 'ppc64le', 'armhf', 'aarch64', 's390x'):
            if getattr(self, 'cc_on_' + arch):
                self.cc_march = arch
                break
        self.cc_name = 'unknown'
        for name in ('gcc', 'clang', 'iccw', 'icc', 'msvc', 'fcc'):
            if getattr(self, 'cc_is_' + name):
                self.cc_name = name
                break
        self.cc_flags = {}
        compiler_flags = self.conf_cc_flags.get(self.cc_name)
        if compiler_flags is None:
            self.dist_fatal("undefined flag for compiler '%s', leave an empty dict instead" % self.cc_name)
        for name, flags in compiler_flags.items():
            self.cc_flags[name] = nflags = []
            if flags:
                assert isinstance(flags, str)
                flags = flags.split()
                for f in flags:
                    if self.cc_test_flags([f]):
                        nflags.append(f)
        self.cc_is_cached = True

    @_Cache.me
    def cc_test_flags(self, flags):
        """
        Returns True if the compiler supports 'flags'.
        """
        assert isinstance(flags, list)
        self.dist_log('testing flags', flags)
        test_path = os.path.join(self.conf_check_path, 'test_flags.c')
        test = self.dist_test(test_path, flags)
        if not test:
            self.dist_log('testing failed', stderr=True)
        return test

    @_Cache.me
    def cc_test_cexpr(self, cexpr, flags=[]):
        """
        Same as the above but supports compile-time expressions.
        """
        self.dist_log('testing compiler expression', cexpr)
        test_path = os.path.join(self.conf_tmp_path, 'npy_dist_test_cexpr.c')
        with open(test_path, 'w') as fd:
            fd.write(textwrap.dedent(f'               #if !({cexpr})\n                   #error "unsupported expression"\n               #endif\n               int dummy;\n            '))
        test = self.dist_test(test_path, flags)
        if not test:
            self.dist_log('testing failed', stderr=True)
        return test

    def cc_normalize_flags(self, flags):
        """
        Remove the conflicts that caused due gathering implied features flags.

        Parameters
        ----------
        'flags' list, compiler flags
            flags should be sorted from the lowest to the highest interest.

        Returns
        -------
        list, filtered from any conflicts.

        Examples
        --------
        >>> self.cc_normalize_flags(['-march=armv8.2-a+fp16', '-march=armv8.2-a+dotprod'])
        ['armv8.2-a+fp16+dotprod']

        >>> self.cc_normalize_flags(
            ['-msse', '-msse2', '-msse3', '-mssse3', '-msse4.1', '-msse4.2', '-mavx', '-march=core-avx2']
        )
        ['-march=core-avx2']
        """
        assert isinstance(flags, list)
        if self.cc_is_gcc or self.cc_is_clang or self.cc_is_icc:
            return self._cc_normalize_unix(flags)
        if self.cc_is_msvc or self.cc_is_iccw:
            return self._cc_normalize_win(flags)
        return flags
    _cc_normalize_unix_mrgx = re.compile('^(-mcpu=|-march=|-x[A-Z0-9\\-])')
    _cc_normalize_unix_frgx = re.compile('^(?!(-mcpu=|-march=|-x[A-Z0-9\\-]|-m[a-z0-9\\-\\.]*.$))|(?:-mzvector)')
    _cc_normalize_unix_krgx = re.compile('^(-mfpu|-mtune)')
    _cc_normalize_arch_ver = re.compile('[0-9.]')

    def _cc_normalize_unix(self, flags):

        def ver_flags(f):
            tokens = f.split('+')
            ver = float('0' + ''.join(re.findall(self._cc_normalize_arch_ver, tokens[0])))
            return (ver, tokens[0], tokens[1:])
        if len(flags) <= 1:
            return flags
        for i, cur_flag in enumerate(reversed(flags)):
            if not re.match(self._cc_normalize_unix_mrgx, cur_flag):
                continue
            lower_flags = flags[:-(i + 1)]
            upper_flags = flags[-i:]
            filtered = list(filter(self._cc_normalize_unix_frgx.search, lower_flags))
            ver, arch, subflags = ver_flags(cur_flag)
            if ver > 0 and len(subflags) > 0:
                for xflag in lower_flags:
                    xver, _, xsubflags = ver_flags(xflag)
                    if ver == xver:
                        subflags = xsubflags + subflags
                cur_flag = arch + '+' + '+'.join(subflags)
            flags = filtered + [cur_flag]
            if i > 0:
                flags += upper_flags
            break
        final_flags = []
        matched = set()
        for f in reversed(flags):
            match = re.match(self._cc_normalize_unix_krgx, f)
            if not match:
                pass
            elif match[0] in matched:
                continue
            else:
                matched.add(match[0])
            final_flags.insert(0, f)
        return final_flags
    _cc_normalize_win_frgx = re.compile('^(?!(/arch\\:|/Qx\\:))')
    _cc_normalize_win_mrgx = re.compile('^(/arch|/Qx:)')

    def _cc_normalize_win(self, flags):
        for i, f in enumerate(reversed(flags)):
            if not re.match(self._cc_normalize_win_mrgx, f):
                continue
            i += 1
            return list(filter(self._cc_normalize_win_frgx.search, flags[:-i])) + flags[-i:]
        return flags