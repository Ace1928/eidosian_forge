from collections import namedtuple
import textwrap
def _validate_bool(value):
    if not isinstance(value, bool):
        raise TypeError('Expected bool value, got {0}'.format(type(value)))