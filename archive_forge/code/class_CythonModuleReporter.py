from __future__ import absolute_import
import re
import os.path
import sys
from collections import defaultdict
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter  # requires coverage.py 4.0+
from coverage.files import canonical_filename
from .Utils import find_root_package_dir, is_package_dir, is_cython_generated_file, open_source_file
from . import __version__
class CythonModuleReporter(FileReporter):
    """
    Provide detailed trace information for one source file to coverage.py.
    """

    def __init__(self, c_file, source_file, rel_file_path, code, excluded_lines):
        super(CythonModuleReporter, self).__init__(source_file)
        self.name = rel_file_path
        self.c_file = c_file
        self._code = code
        self._excluded_lines = excluded_lines

    def lines(self):
        """
        Return set of line numbers that are possibly executable.
        """
        return set(self._code)

    def excluded_lines(self):
        """
        Return set of line numbers that are excluded from coverage.
        """
        return self._excluded_lines

    def _iter_source_tokens(self):
        current_line = 1
        for line_no, code_line in sorted(self._code.items()):
            while line_no > current_line:
                yield []
                current_line += 1
            yield [('txt', code_line)]
            current_line += 1

    def source(self):
        """
        Return the source code of the file as a string.
        """
        if os.path.exists(self.filename):
            with open_source_file(self.filename) as f:
                return f.read()
        else:
            return '\n'.join((tokens[0][1] if tokens else '' for tokens in self._iter_source_tokens()))

    def source_token_lines(self):
        """
        Iterate over the source code tokens.
        """
        if os.path.exists(self.filename):
            with open_source_file(self.filename) as f:
                for line in f:
                    yield [('txt', line.rstrip('\n'))]
        else:
            for line in self._iter_source_tokens():
                yield [('txt', line)]