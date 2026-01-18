from collections import namedtuple
import textwrap
def _validate_display_precision(value):
    if value is not None:
        if not isinstance(value, int) or not 0 <= value <= 16:
            raise ValueError('Invalid value, needs to be an integer [0-16]')