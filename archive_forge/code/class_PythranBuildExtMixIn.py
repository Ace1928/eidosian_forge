import pythran.config as cfg
from collections import defaultdict
import os.path
import os
class PythranBuildExtMixIn(object):
    """Subclass of `distutils.command.build_ext.build_ext` which is required to
    build `PythranExtension` with the configured C++ compiler. It may also be
    subclassed if you want to combine with another build_ext class (NumPy,
    Cython implementations).

    """

    def build_extension(self, ext):
        StringTypes = (str,)

        def get_value(obj, key):
            var = getattr(obj, key)
            if isinstance(var, Iterable) and (not isinstance(var, StringTypes)):
                return var[0]
            else:
                return var

        def set_value(obj, key, value):
            var = getattr(obj, key)
            if isinstance(var, Iterable) and (not isinstance(var, StringTypes)):
                var[0] = value
            else:
                setattr(obj, key, value)
        prev = {'preprocessor': None, 'compiler_cxx': None, 'compiler_so': None, 'compiler': None, 'linker_exe': None, 'linker_so': None, 'cc': None}
        for key in list(prev.keys()):
            if hasattr(self.compiler, key):
                prev[key] = get_value(self.compiler, key)
            else:
                del prev[key]
        if getattr(ext, 'cxx', None) is not None:
            for comp in prev:
                if hasattr(self.compiler, comp):
                    set_value(self.compiler, comp, ext.cxx)
        find_exe = None
        if getattr(ext, 'cc', None) is not None:
            try:
                import distutils._msvccompiler as msvc
                find_exe = msvc._find_exe

                def _find_exe(exe, *args, **kwargs):
                    if exe == 'cl.exe':
                        exe = ext.cc
                    return find_exe(exe, *args, **kwargs)
                msvc._find_exe = _find_exe
            except ImportError:
                pass
        for flag in cfg.cfg.get('compiler', 'ignoreflags').split():
            for target in ('compiler_so', 'linker_so'):
                try:
                    while True:
                        getattr(self.compiler, target).remove(flag)
                except (AttributeError, ValueError):
                    pass
        if hasattr(self.compiler, 'compiler_so'):
            archs = defaultdict(list)
            for i, flag in enumerate(self.compiler.compiler_so[1:]):
                if self.compiler.compiler_so[i] == '-arch':
                    archs[flag].append(i + 1)
            if 'x86_64' in archs and 'i386' in archs:
                for i in archs['i386']:
                    self.compiler.compiler_so[i] = 'x86_64'
        try:
            return super(PythranBuildExtMixIn, self).build_extension(ext)
        finally:
            for key in prev.keys():
                set_value(self.compiler, key, prev[key])
            if find_exe is not None:
                import distutils._msvccompiler as msvc
                msvc._find_exe = find_exe