import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def add_data_files(self, *files):
    """Add data files to configuration data_files.

        Parameters
        ----------
        files : sequence
            Argument(s) can be either

                * 2-sequence (<datadir prefix>,<path to data file(s)>)
                * paths to data files where python datadir prefix defaults
                  to package dir.

        Notes
        -----
        The form of each element of the files sequence is very flexible
        allowing many combinations of where to get the files from the package
        and where they should ultimately be installed on the system. The most
        basic usage is for an element of the files argument sequence to be a
        simple filename. This will cause that file from the local path to be
        installed to the installation path of the self.name package (package
        path). The file argument can also be a relative path in which case the
        entire relative path will be installed into the package directory.
        Finally, the file can be an absolute path name in which case the file
        will be found at the absolute path name but installed to the package
        path.

        This basic behavior can be augmented by passing a 2-tuple in as the
        file argument. The first element of the tuple should specify the
        relative path (under the package install directory) where the
        remaining sequence of files should be installed to (it has nothing to
        do with the file-names in the source distribution). The second element
        of the tuple is the sequence of files that should be installed. The
        files in this sequence can be filenames, relative paths, or absolute
        paths. For absolute paths the file will be installed in the top-level
        package installation directory (regardless of the first argument).
        Filenames and relative path names will be installed in the package
        install directory under the path name given as the first element of
        the tuple.

        Rules for installation paths:

          #. file.txt -> (., file.txt)-> parent/file.txt
          #. foo/file.txt -> (foo, foo/file.txt) -> parent/foo/file.txt
          #. /foo/bar/file.txt -> (., /foo/bar/file.txt) -> parent/file.txt
          #. ``*``.txt -> parent/a.txt, parent/b.txt
          #. foo/``*``.txt`` -> parent/foo/a.txt, parent/foo/b.txt
          #. ``*/*.txt`` -> (``*``, ``*``/``*``.txt) -> parent/c/a.txt, parent/d/b.txt
          #. (sun, file.txt) -> parent/sun/file.txt
          #. (sun, bar/file.txt) -> parent/sun/file.txt
          #. (sun, /foo/bar/file.txt) -> parent/sun/file.txt
          #. (sun, ``*``.txt) -> parent/sun/a.txt, parent/sun/b.txt
          #. (sun, bar/``*``.txt) -> parent/sun/a.txt, parent/sun/b.txt
          #. (sun/``*``, ``*``/``*``.txt) -> parent/sun/c/a.txt, parent/d/b.txt

        An additional feature is that the path to a data-file can actually be
        a function that takes no arguments and returns the actual path(s) to
        the data-files. This is useful when the data files are generated while
        building the package.

        Examples
        --------
        Add files to the list of data_files to be included with the package.

            >>> self.add_data_files('foo.dat',
            ...     ('fun', ['gun.dat', 'nun/pun.dat', '/tmp/sun.dat']),
            ...     'bar/cat.dat',
            ...     '/full/path/to/can.dat')                   #doctest: +SKIP

        will install these data files to::

            <package install directory>/
             foo.dat
             fun/
               gun.dat
               nun/
                 pun.dat
             sun.dat
             bar/
               car.dat
             can.dat

        where <package install directory> is the package (or sub-package)
        directory such as '/usr/lib/python2.4/site-packages/mypackage' ('C:
        \\Python2.4 \\Lib \\site-packages \\mypackage') or
        '/usr/lib/python2.4/site- packages/mypackage/mysubpackage' ('C:
        \\Python2.4 \\Lib \\site-packages \\mypackage \\mysubpackage').
        """
    if len(files) > 1:
        for f in files:
            self.add_data_files(f)
        return
    assert len(files) == 1
    if is_sequence(files[0]):
        d, files = files[0]
    else:
        d = None
    if is_string(files):
        filepat = files
    elif is_sequence(files):
        if len(files) == 1:
            filepat = files[0]
        else:
            for f in files:
                self.add_data_files((d, f))
            return
    else:
        raise TypeError(repr(type(files)))
    if d is None:
        if hasattr(filepat, '__call__'):
            d = ''
        elif os.path.isabs(filepat):
            d = ''
        else:
            d = os.path.dirname(filepat)
        self.add_data_files((d, files))
        return
    paths = self.paths(filepat, include_non_existing=False)
    if is_glob_pattern(filepat):
        if is_glob_pattern(d):
            pattern_list = d.split(os.sep)
            pattern_list.reverse()
            for path in paths:
                path_list = path.split(os.sep)
                path_list.reverse()
                path_list.pop()
                target_list = []
                i = 0
                for s in pattern_list:
                    if is_glob_pattern(s):
                        target_list.append(path_list[i])
                        i += 1
                    else:
                        target_list.append(s)
                target_list.reverse()
                self.add_data_files((os.sep.join(target_list), path))
        else:
            self.add_data_files((d, paths))
        return
    assert not is_glob_pattern(d), repr((d, filepat))
    dist = self.get_distribution()
    if dist is not None and dist.data_files is not None:
        data_files = dist.data_files
    else:
        data_files = self.data_files
    data_files.append((os.path.join(self.path_in_package, d), paths))