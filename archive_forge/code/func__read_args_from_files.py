import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _read_args_from_files(self, arg_strings):
    new_arg_strings = []
    for arg_string in arg_strings:
        if arg_string[0] not in self.fromfile_prefix_chars:
            new_arg_strings.append(arg_string)
        else:
            try:
                args_file = open(arg_string[1:])
                try:
                    arg_strings = []
                    for arg_line in args_file.read().splitlines():
                        for arg in self.convert_arg_line_to_args(arg_line):
                            arg_strings.append(arg)
                    arg_strings = self._read_args_from_files(arg_strings)
                    new_arg_strings.extend(arg_strings)
                finally:
                    args_file.close()
            except IOError:
                err = _sys.exc_info()[1]
                self.error(str(err))
    return new_arg_strings