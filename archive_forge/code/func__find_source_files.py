from __future__ import absolute_import
import re
import os.path
import sys
from collections import defaultdict
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter  # requires coverage.py 4.0+
from coverage.files import canonical_filename
from .Utils import find_root_package_dir, is_package_dir, is_cython_generated_file, open_source_file
from . import __version__
def _find_source_files(self, filename):
    basename, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext in MODULE_FILE_EXTENSIONS:
        pass
    elif ext == '.pyd':
        platform_suffix = re.search('[.]cp[0-9]+-win[_a-z0-9]*$', basename, re.I)
        if platform_suffix:
            basename = basename[:platform_suffix.start()]
    elif ext == '.so':
        platform_suffix = re.search('[.](?:cpython|pypy)-[0-9]+[-_a-z0-9]*$', basename, re.I)
        if platform_suffix:
            basename = basename[:platform_suffix.start()]
    elif ext == '.pxi':
        self._find_c_source_files(os.path.dirname(filename), filename)
        if filename in self._c_files_map:
            return (self._c_files_map[filename][0], None)
    else:
        return (None, None)
    c_file = filename if ext in C_FILE_EXTENSIONS else _find_c_source(basename)
    if c_file is None:
        package_root = find_root_package_dir.uncached(filename)
        package_path = os.path.relpath(basename, package_root).split(os.path.sep)
        if len(package_path) > 1:
            test_basepath = os.path.join(os.path.dirname(filename), '.'.join(package_path))
            c_file = _find_c_source(test_basepath)
    py_source_file = None
    if c_file:
        py_source_file = os.path.splitext(c_file)[0] + '.py'
        if not os.path.exists(py_source_file):
            py_source_file = None
        if not is_cython_generated_file(c_file, if_not_found=False):
            if py_source_file and os.path.exists(c_file):
                py_source_file = None
            c_file = None
    return (c_file, py_source_file)