from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def ConvertToCamelCase(name):
    """Converts snake_case name to camelCase."""
    part = name.split('_')
    return part[0] + ''.join((x.title() for x in part[1:]))