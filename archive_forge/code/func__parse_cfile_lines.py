from __future__ import absolute_import
import re
import os.path
import sys
from collections import defaultdict
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter  # requires coverage.py 4.0+
from coverage.files import canonical_filename
from .Utils import find_root_package_dir, is_package_dir, is_cython_generated_file, open_source_file
from . import __version__
def _parse_cfile_lines(self, c_file):
    """
        Parse a C file and extract all source file lines that generated executable code.
        """
    match_source_path_line = re.compile(' */[*] +"(.*)":([0-9]+)$').match
    match_current_code_line = re.compile(' *[*] (.*) # <<<<<<+$').match
    match_comment_end = re.compile(' *[*]/$').match
    match_trace_line = re.compile(' *__Pyx_TraceLine\\(([0-9]+),').match
    not_executable = re.compile('\\s*c(?:type)?def\\s+(?:(?:public|external)\\s+)?(?:struct|union|enum|class)(\\s+[^:]+|)\\s*:').match
    if self._excluded_line_patterns:
        line_is_excluded = re.compile('|'.join(['(?:%s)' % regex for regex in self._excluded_line_patterns])).search
    else:
        line_is_excluded = lambda line: False
    code_lines = defaultdict(dict)
    executable_lines = defaultdict(set)
    current_filename = None
    if self._excluded_lines_map is None:
        self._excluded_lines_map = defaultdict(set)
    with open(c_file) as lines:
        lines = iter(lines)
        for line in lines:
            match = match_source_path_line(line)
            if not match:
                if '__Pyx_TraceLine(' in line and current_filename is not None:
                    trace_line = match_trace_line(line)
                    if trace_line:
                        executable_lines[current_filename].add(int(trace_line.group(1)))
                continue
            filename, lineno = match.groups()
            current_filename = filename
            lineno = int(lineno)
            for comment_line in lines:
                match = match_current_code_line(comment_line)
                if match:
                    code_line = match.group(1).rstrip()
                    if not_executable(code_line):
                        break
                    if line_is_excluded(code_line):
                        self._excluded_lines_map[filename].add(lineno)
                        break
                    code_lines[filename][lineno] = code_line
                    break
                elif match_comment_end(comment_line):
                    break
    for filename, lines in code_lines.items():
        dead_lines = set(lines).difference(executable_lines.get(filename, ()))
        for lineno in dead_lines:
            del lines[lineno]
    return code_lines