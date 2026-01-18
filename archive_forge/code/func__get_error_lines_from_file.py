from __future__ import (absolute_import, division, print_function)
import re
import traceback
from collections.abc import Sequence
from ansible.errors.yaml_strings import (
from ansible.module_utils.common.text.converters import to_native, to_text
def _get_error_lines_from_file(self, file_name, line_number):
    """
        Returns the line in the file which corresponds to the reported error
        location, as well as the line preceding it (if the error did not
        occur on the first line), to provide context to the error.
        """
    target_line = ''
    prev_line = ''
    with open(file_name, 'r') as f:
        lines = f.readlines()
        file_length = len(lines)
        if line_number >= file_length:
            line_number = file_length - 1
        target_line = lines[line_number]
        while not target_line.strip():
            line_number -= 1
            target_line = lines[line_number]
        if line_number > 0:
            prev_line = lines[line_number - 1]
    return (target_line, prev_line)