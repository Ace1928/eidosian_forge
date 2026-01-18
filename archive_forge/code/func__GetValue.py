from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _GetValue(index):
    return _GetMetaDataValue(resource['items'], index, deserialize=bool(key))