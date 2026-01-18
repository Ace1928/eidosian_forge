from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def FlagNameFormat(arg_name):
    """Format a string as a flag name."""
    return PREFIX + KebabCase(StripPrefix(arg_name)).lower()