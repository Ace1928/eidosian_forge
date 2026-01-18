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
class PyZipFile(ZipFile):
    """Class to create ZIP archives with Python library files and packages."""

    def __init__(self, file, mode='r', compression=ZIP_STORED, allowZip64=True, optimize=-1):
        ZipFile.__init__(self, file, mode=mode, compression=compression, allowZip64=allowZip64)
        self._optimize = optimize

    def writepy(self, pathname, basename='', filterfunc=None):
        """Add all files from "pathname" to the ZIP archive.

        If pathname is a package directory, search the directory and
        all package subdirectories recursively for all *.py and enter
        the modules into the archive.  If pathname is a plain
        directory, listdir *.py and enter all modules.  Else, pathname
        must be a Python *.py file and the module will be put into the
        archive.  Added modules are always module.pyc.
        This method will compile the module.py into module.pyc if
        necessary.
        If filterfunc(pathname) is given, it is called with every argument.
        When it is False, the file or directory is skipped.
        """
        pathname = os.fspath(pathname)
        if filterfunc and (not filterfunc(pathname)):
            if self.debug:
                label = 'path' if os.path.isdir(pathname) else 'file'
                print('%s %r skipped by filterfunc' % (label, pathname))
            return
        dir, name = os.path.split(pathname)
        if os.path.isdir(pathname):
            initname = os.path.join(pathname, '__init__.py')
            if os.path.isfile(initname):
                if basename:
                    basename = '%s/%s' % (basename, name)
                else:
                    basename = name
                if self.debug:
                    print('Adding package in', pathname, 'as', basename)
                fname, arcname = self._get_codename(initname[0:-3], basename)
                if self.debug:
                    print('Adding', arcname)
                self.write(fname, arcname)
                dirlist = sorted(os.listdir(pathname))
                dirlist.remove('__init__.py')
                for filename in dirlist:
                    path = os.path.join(pathname, filename)
                    root, ext = os.path.splitext(filename)
                    if os.path.isdir(path):
                        if os.path.isfile(os.path.join(path, '__init__.py')):
                            self.writepy(path, basename, filterfunc=filterfunc)
                    elif ext == '.py':
                        if filterfunc and (not filterfunc(path)):
                            if self.debug:
                                print('file %r skipped by filterfunc' % path)
                            continue
                        fname, arcname = self._get_codename(path[0:-3], basename)
                        if self.debug:
                            print('Adding', arcname)
                        self.write(fname, arcname)
            else:
                if self.debug:
                    print('Adding files from directory', pathname)
                for filename in sorted(os.listdir(pathname)):
                    path = os.path.join(pathname, filename)
                    root, ext = os.path.splitext(filename)
                    if ext == '.py':
                        if filterfunc and (not filterfunc(path)):
                            if self.debug:
                                print('file %r skipped by filterfunc' % path)
                            continue
                        fname, arcname = self._get_codename(path[0:-3], basename)
                        if self.debug:
                            print('Adding', arcname)
                        self.write(fname, arcname)
        else:
            if pathname[-3:] != '.py':
                raise RuntimeError('Files added with writepy() must end with ".py"')
            fname, arcname = self._get_codename(pathname[0:-3], basename)
            if self.debug:
                print('Adding file', arcname)
            self.write(fname, arcname)

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