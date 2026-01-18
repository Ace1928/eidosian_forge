from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def ConvertToAngrySnakeCase(name):
    """Converts camelCase name to ANGRY_SNAKE_CASE."""
    return _SNAKE_RE.sub('_\\1', name).upper()