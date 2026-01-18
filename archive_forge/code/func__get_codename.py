import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def _get_codename(self, pathname, basename):
    """Return (filename, archivename) for the path.

        Given a module name path, return the correct file path and
        archive name, compiling if necessary.  For example, given
        /python/lib/string, return (/python/lib/string.pyc, string).
        """

    def _compile(file, optimize=-1):
        import py_compile
        if self.debug:
            print('Compiling', file)
        try:
            py_compile.compile(file, doraise=True, optimize=optimize)
        except py_compile.PyCompileError as err:
            print(err.msg)
            return False
        return True
    file_py = pathname + '.py'
    file_pyc = pathname + '.pyc'
    pycache_opt0 = importlib.util.cache_from_source(file_py, optimization='')
    pycache_opt1 = importlib.util.cache_from_source(file_py, optimization=1)
    pycache_opt2 = importlib.util.cache_from_source(file_py, optimization=2)
    if self._optimize == -1:
        if os.path.isfile(file_pyc) and os.stat(file_pyc).st_mtime >= os.stat(file_py).st_mtime:
            arcname = fname = file_pyc
        elif os.path.isfile(pycache_opt0) and os.stat(pycache_opt0).st_mtime >= os.stat(file_py).st_mtime:
            fname = pycache_opt0
            arcname = file_pyc
        elif os.path.isfile(pycache_opt1) and os.stat(pycache_opt1).st_mtime >= os.stat(file_py).st_mtime:
            fname = pycache_opt1
            arcname = file_pyc
        elif os.path.isfile(pycache_opt2) and os.stat(pycache_opt2).st_mtime >= os.stat(file_py).st_mtime:
            fname = pycache_opt2
            arcname = file_pyc
        elif _compile(file_py):
            if sys.flags.optimize == 0:
                fname = pycache_opt0
            elif sys.flags.optimize == 1:
                fname = pycache_opt1
            else:
                fname = pycache_opt2
            arcname = file_pyc
        else:
            fname = arcname = file_py
    else:
        if self._optimize == 0:
            fname = pycache_opt0
            arcname = file_pyc
        else:
            arcname = file_pyc
            if self._optimize == 1:
                fname = pycache_opt1
            elif self._optimize == 2:
                fname = pycache_opt2
            else:
                msg = "invalid value for 'optimize': {!r}".format(self._optimize)
                raise ValueError(msg)
        if not (os.path.isfile(fname) and os.stat(fname).st_mtime >= os.stat(file_py).st_mtime):
            if not _compile(file_py, optimize=self._optimize):
                fname = arcname = file_py
    archivename = os.path.split(arcname)[1]
    if basename:
        archivename = '%s/%s' % (basename, archivename)
    return (fname, archivename)