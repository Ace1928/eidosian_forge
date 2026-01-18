from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _consume_google_args_line(line_info, state):
    """Consume a single line from a Google args section."""
    split_line = line_info.remaining.split(':', 1)
    if len(split_line) > 1:
        first, second = split_line
        if _is_arg_name(first.strip()):
            arg = _get_or_create_arg_by_name(state, first.strip())
            arg.description.lines.append(second.strip())
            state.current_arg = arg
        else:
            arg_name_and_type = _as_arg_name_and_type(first)
            if arg_name_and_type:
                arg_name, type_str = arg_name_and_type
                arg = _get_or_create_arg_by_name(state, arg_name)
                arg.type.lines.append(type_str)
                arg.description.lines.append(second.strip())
                state.current_arg = arg
            elif state.current_arg:
                state.current_arg.description.lines.append(split_line[0])
    elif state.current_arg:
        state.current_arg.description.lines.append(split_line[0])