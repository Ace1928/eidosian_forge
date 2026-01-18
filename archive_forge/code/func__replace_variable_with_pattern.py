from __future__ import unicode_literals
from collections import deque
import copy
import functools
import re
def _replace_variable_with_pattern(match):
    """Replace a variable match with a pattern that can be used to validate it.

    Args:
        match (re.Match): A regular expression match

    Returns:
        str: A regular expression pattern that can be used to validate the
            variable in an expanded path.

    Raises:
        ValueError: If an unexpected template expression is encountered.
    """
    positional = match.group('positional')
    name = match.group('name')
    template = match.group('template')
    if name is not None:
        if not template:
            return _SINGLE_SEGMENT_PATTERN.format(name)
        elif template == '**':
            return _MULTI_SEGMENT_PATTERN.format(name)
        else:
            return _generate_pattern_for_template(template)
    elif positional == '*':
        return _SINGLE_SEGMENT_PATTERN
    elif positional == '**':
        return _MULTI_SEGMENT_PATTERN
    else:
        raise ValueError('Unknown template expression {}'.format(match.group(0)))