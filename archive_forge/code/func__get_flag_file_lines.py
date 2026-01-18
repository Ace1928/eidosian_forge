from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def _get_flag_file_lines(self, filename, parsed_file_stack=None):
    """Returns the useful (!=comments, etc) lines from a file with flags.

    Args:
      filename: str, the name of the flag file.
      parsed_file_stack: [str], a list of the names of the files that we have
        recursively encountered at the current depth. MUTATED BY THIS FUNCTION
        (but the original value is preserved upon successfully returning from
        function call).

    Returns:
      List of strings. See the note below.

    NOTE(springer): This function checks for a nested --flagfile=<foo>
    tag and handles the lower file recursively. It returns a list of
    all the lines that _could_ contain command flags. This is
    EVERYTHING except whitespace lines and comments (lines starting
    with '#' or '//').
    """
    if not filename:
        return []
    if parsed_file_stack is None:
        parsed_file_stack = []
    if filename in parsed_file_stack:
        sys.stderr.write('Warning: Hit circular flagfile dependency. Ignoring flagfile: %s\n' % (filename,))
        return []
    else:
        parsed_file_stack.append(filename)
    line_list = []
    flag_line_list = []
    try:
        file_obj = open(filename, 'r')
    except IOError as e_msg:
        raise _exceptions.CantOpenFlagFileError('ERROR:: Unable to open flagfile: %s' % e_msg)
    with file_obj:
        line_list = file_obj.readlines()
    for line in line_list:
        if line.isspace():
            pass
        elif line.startswith('#') or line.startswith('//'):
            pass
        elif self._is_flag_file_directive(line):
            sub_filename = self._extract_filename(line)
            included_flags = self._get_flag_file_lines(sub_filename, parsed_file_stack=parsed_file_stack)
            flag_line_list.extend(included_flags)
        else:
            flag_line_list.append(line.strip())
    parsed_file_stack.pop()
    return flag_line_list