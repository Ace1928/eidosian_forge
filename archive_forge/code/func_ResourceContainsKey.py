from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def ResourceContainsKey(resource, key):
    """True if resource contains key, else False."""
    return Get(resource, key, None) is not None