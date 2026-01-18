from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import OrderedDict
import json
import re
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core.util import files
import six
def _ExtractCollectionName(method_id):
    """Extract the name of the collection from a method ID."""
    match = _METHOD_ID_RE.match(method_id)
    if match:
        return match.group('collection')
    else:
        raise NoMatchingMethodError('Method {0} does not match regexp {1}.'.format(method_id, _METHOD_ID_RE_RAW))