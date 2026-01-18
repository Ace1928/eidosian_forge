import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
class _Distutils:
    """A helper class that provides a collection of fundamental methods
    implemented in a top of Python and NumPy Distutils.

    The idea behind this class is to gather all methods that it may
    need to override in case of reuse 'CCompilerOpt' in environment
    different than of what NumPy has.

    Parameters
    ----------
    ccompiler : `CCompiler`
        The generate instance that returned from `distutils.ccompiler.new_compiler()`.
    """

    def __init__(self, ccompiler):
        self._ccompiler = ccompiler

    def dist_compile(self, sources, flags, ccompiler=None, **kwargs):
        """Wrap CCompiler.compile()"""
        assert isinstance(sources, list)
        assert isinstance(flags, list)
        flags = kwargs.pop('extra_postargs', []) + flags
        if not ccompiler:
            ccompiler = self._ccompiler
        return ccompiler.compile(sources, extra_postargs=flags, **kwargs)

    def dist_test(self, source, flags, macros=[]):
        """Return True if 'CCompiler.compile()' able to compile
        a source file with certain flags.
        """
        assert isinstance(source, str)
        from distutils.errors import CompileError
        cc = self._ccompiler
        bk_spawn = getattr(cc, 'spawn', None)
        if bk_spawn:
            cc_type = getattr(self._ccompiler, 'compiler_type', '')
            if cc_type in ('msvc',):
                setattr(cc, 'spawn', self._dist_test_spawn_paths)
            else:
                setattr(cc, 'spawn', self._dist_test_spawn)
        test = False
        try:
            self.dist_compile([source], flags, macros=macros, output_dir=self.conf_tmp_path)
            test = True
        except CompileError as e:
            self.dist_log(str(e), stderr=True)
        if bk_spawn:
            setattr(cc, 'spawn', bk_spawn)
        return test

    def dist_info(self):
        """
        Return a tuple containing info about (platform, compiler, extra_args),
        required by the abstract class '_CCompiler' for discovering the
        platform environment. This is also used as a cache factor in order
        to detect any changes happening from outside.
        """
        if hasattr(self, '_dist_info'):
            return self._dist_info
        cc_type = getattr(self._ccompiler, 'compiler_type', '')
        if cc_type in ('intelem', 'intelemw'):
            platform = 'x86_64'
        elif cc_type in ('intel', 'intelw', 'intele'):
            platform = 'x86'
        else:
            from distutils.util import get_platform
            platform = get_platform()
        cc_info = getattr(self._ccompiler, 'compiler', getattr(self._ccompiler, 'compiler_so', ''))
        if not cc_type or cc_type == 'unix':
            if hasattr(cc_info, '__iter__'):
                compiler = cc_info[0]
            else:
                compiler = str(cc_info)
        else:
            compiler = cc_type
        if hasattr(cc_info, '__iter__') and len(cc_info) > 1:
            extra_args = ' '.join(cc_info[1:])
        else:
            extra_args = os.environ.get('CFLAGS', '')
            extra_args += os.environ.get('CPPFLAGS', '')
        self._dist_info = (platform, compiler, extra_args)
        return self._dist_info

    @staticmethod
    def dist_error(*args):
        """Raise a compiler error"""
        from distutils.errors import CompileError
        raise CompileError(_Distutils._dist_str(*args))

    @staticmethod
    def dist_fatal(*args):
        """Raise a distutils error"""
        from distutils.errors import DistutilsError
        raise DistutilsError(_Distutils._dist_str(*args))

    @staticmethod
    def dist_log(*args, stderr=False):
        """Print a console message"""
        from numpy.distutils import log
        out = _Distutils._dist_str(*args)
        if stderr:
            log.warn(out)
        else:
            log.info(out)

    @staticmethod
    def dist_load_module(name, path):
        """Load a module from file, required by the abstract class '_Cache'."""
        from .misc_util import exec_mod_from_location
        try:
            return exec_mod_from_location(name, path)
        except Exception as e:
            _Distutils.dist_log(e, stderr=True)
        return None

    @staticmethod
    def _dist_str(*args):
        """Return a string to print by log and errors."""

        def to_str(arg):
            if not isinstance(arg, str) and hasattr(arg, '__iter__'):
                ret = []
                for a in arg:
                    ret.append(to_str(a))
                return '(' + ' '.join(ret) + ')'
            return str(arg)
        stack = inspect.stack()[2]
        start = 'CCompilerOpt.%s[%d] : ' % (stack.function, stack.lineno)
        out = ' '.join([to_str(a) for a in (*args,)])
        return start + out

    def _dist_test_spawn_paths(self, cmd, display=None):
        """
        Fix msvc SDK ENV path same as distutils do
        without it we get c1: fatal error C1356: unable to find mspdbcore.dll
        """
        if not hasattr(self._ccompiler, '_paths'):
            self._dist_test_spawn(cmd)
            return
        old_path = os.getenv('path')
        try:
            os.environ['path'] = self._ccompiler._paths
            self._dist_test_spawn(cmd)
        finally:
            os.environ['path'] = old_path
    _dist_warn_regex = re.compile('.*(warning D9002|invalid argument for option).*')

    @staticmethod
    def _dist_test_spawn(cmd, display=None):
        try:
            o = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            if o and re.match(_Distutils._dist_warn_regex, o):
                _Distutils.dist_error('Flags in command', cmd, "aren't supported by the compiler, output -> \n%s" % o)
        except subprocess.CalledProcessError as exc:
            o = exc.output
            s = exc.returncode
        except OSError as e:
            o = e
            s = 127
        else:
            return None
        _Distutils.dist_error('Command', cmd, 'failed with exit status %d output -> \n%s' % (s, o))