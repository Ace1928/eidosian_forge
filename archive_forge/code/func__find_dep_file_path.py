from __future__ import absolute_import
import re
import os.path
import sys
from collections import defaultdict
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter  # requires coverage.py 4.0+
from coverage.files import canonical_filename
from .Utils import find_root_package_dir, is_package_dir, is_cython_generated_file, open_source_file
from . import __version__
def _find_dep_file_path(main_file, file_path, relative_path_search=False):
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path) and (file_path.endswith('.pxi') or relative_path_search):
        rel_file_path = os.path.join(os.path.dirname(main_file), file_path)
        if os.path.exists(rel_file_path):
            abs_path = os.path.abspath(rel_file_path)
        abs_no_ext = os.path.splitext(abs_path)[0]
        file_no_ext, extension = os.path.splitext(file_path)
        abs_no_ext = os.path.normpath(abs_no_ext)
        file_no_ext = os.path.normpath(file_no_ext)
        matching_paths = zip(reversed(abs_no_ext.split(os.sep)), reversed(file_no_ext.split(os.sep)))
        for one, other in matching_paths:
            if one != other:
                break
        else:
            matching_abs_path = os.path.splitext(main_file)[0] + extension
            if os.path.exists(matching_abs_path):
                return canonical_filename(matching_abs_path)
    if not os.path.exists(abs_path):
        for sys_path in sys.path:
            test_path = os.path.realpath(os.path.join(sys_path, file_path))
            if os.path.exists(test_path):
                return canonical_filename(test_path)
    return canonical_filename(abs_path)