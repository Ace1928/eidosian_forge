from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def PositionalFormat(arg_name):
    """Format a string as a positional."""
    return SnakeCase(StripPrefix(arg_name)).upper()