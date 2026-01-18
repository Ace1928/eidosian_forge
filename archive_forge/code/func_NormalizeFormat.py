from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def NormalizeFormat(arg_name):
    """Converts arg name to lower snake case, no '--' prefix."""
    return SnakeCase(StripPrefix(arg_name)).lower()