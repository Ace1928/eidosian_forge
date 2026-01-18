import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def _apply_embedding_fix(self, kwds):

    def ensure(key, value):
        lst = kwds.setdefault(key, [])
        if value not in lst:
            lst.append(value)
    if '__pypy__' in sys.builtin_module_names:
        import os
        if sys.platform == 'win32':
            pythonlib = 'python{0[0]}{0[1]}'.format(sys.version_info)
            if hasattr(sys, 'prefix'):
                ensure('library_dirs', os.path.join(sys.prefix, 'libs'))
        else:
            if sys.version_info < (3,):
                pythonlib = 'pypy-c'
            else:
                pythonlib = 'pypy3-c'
            if hasattr(sys, 'prefix'):
                ensure('library_dirs', os.path.join(sys.prefix, 'bin'))
        if hasattr(sys, 'prefix'):
            ensure('library_dirs', os.path.join(sys.prefix, 'pypy', 'goal'))
    else:
        if sys.platform == 'win32':
            template = 'python%d%d'
            if hasattr(sys, 'gettotalrefcount'):
                template += '_d'
        else:
            try:
                import sysconfig
            except ImportError:
                from cffi._shimmed_dist_utils import sysconfig
            template = 'python%d.%d'
            if sysconfig.get_config_var('DEBUG_EXT'):
                template += sysconfig.get_config_var('DEBUG_EXT')
        pythonlib = template % (sys.hexversion >> 24, sys.hexversion >> 16 & 255)
        if hasattr(sys, 'abiflags'):
            pythonlib += sys.abiflags
    ensure('libraries', pythonlib)
    if sys.platform == 'win32':
        ensure('extra_link_args', '/MANIFEST')