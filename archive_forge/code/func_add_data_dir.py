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
def add_data_dir(self, data_path):
    """Recursively add files under data_path to data_files list.

        Recursively add files under data_path to the list of data_files to be
        installed (and distributed). The data_path can be either a relative
        path-name, or an absolute path-name, or a 2-tuple where the first
        argument shows where in the install directory the data directory
        should be installed to.

        Parameters
        ----------
        data_path : seq or str
            Argument can be either

                * 2-sequence (<datadir suffix>, <path to data directory>)
                * path to data directory where python datadir suffix defaults
                  to package dir.

        Notes
        -----
        Rules for installation paths::

            foo/bar -> (foo/bar, foo/bar) -> parent/foo/bar
            (gun, foo/bar) -> parent/gun
            foo/* -> (foo/a, foo/a), (foo/b, foo/b) -> parent/foo/a, parent/foo/b
            (gun, foo/*) -> (gun, foo/a), (gun, foo/b) -> gun
            (gun/*, foo/*) -> parent/gun/a, parent/gun/b
            /foo/bar -> (bar, /foo/bar) -> parent/bar
            (gun, /foo/bar) -> parent/gun
            (fun/*/gun/*, sun/foo/bar) -> parent/fun/foo/gun/bar

        Examples
        --------
        For example suppose the source directory contains fun/foo.dat and
        fun/bar/car.dat:

        >>> self.add_data_dir('fun')                       #doctest: +SKIP
        >>> self.add_data_dir(('sun', 'fun'))              #doctest: +SKIP
        >>> self.add_data_dir(('gun', '/full/path/to/fun'))#doctest: +SKIP

        Will install data-files to the locations::

            <package install directory>/
              fun/
                foo.dat
                bar/
                  car.dat
              sun/
                foo.dat
                bar/
                  car.dat
              gun/
                foo.dat
                car.dat

        """
    if is_sequence(data_path):
        d, data_path = data_path
    else:
        d = None
    if is_sequence(data_path):
        [self.add_data_dir((d, p)) for p in data_path]
        return
    if not is_string(data_path):
        raise TypeError('not a string: %r' % (data_path,))
    if d is None:
        if os.path.isabs(data_path):
            return self.add_data_dir((os.path.basename(data_path), data_path))
        return self.add_data_dir((data_path, data_path))
    paths = self.paths(data_path, include_non_existing=False)
    if is_glob_pattern(data_path):
        if is_glob_pattern(d):
            pattern_list = allpath(d).split(os.sep)
            pattern_list.reverse()
            rl = list(range(len(pattern_list) - 1))
            rl.reverse()
            for i in rl:
                if not pattern_list[i]:
                    del pattern_list[i]
            for path in paths:
                if not os.path.isdir(path):
                    print('Not a directory, skipping', path)
                    continue
                rpath = rel_path(path, self.local_path)
                path_list = rpath.split(os.sep)
                path_list.reverse()
                target_list = []
                i = 0
                for s in pattern_list:
                    if is_glob_pattern(s):
                        if i >= len(path_list):
                            raise ValueError('cannot fill pattern %r with %r' % (d, path))
                        target_list.append(path_list[i])
                    else:
                        assert s == path_list[i], repr((s, path_list[i], data_path, d, path, rpath))
                        target_list.append(s)
                    i += 1
                if path_list[i:]:
                    self.warn('mismatch of pattern_list=%s and path_list=%s' % (pattern_list, path_list))
                target_list.reverse()
                self.add_data_dir((os.sep.join(target_list), path))
        else:
            for path in paths:
                self.add_data_dir((d, path))
        return
    assert not is_glob_pattern(d), repr(d)
    dist = self.get_distribution()
    if dist is not None and dist.data_files is not None:
        data_files = dist.data_files
    else:
        data_files = self.data_files
    for path in paths:
        for d1, f in list(general_source_directories_files(path)):
            target_path = os.path.join(self.path_in_package, d, d1)
            data_files.append((target_path, f))