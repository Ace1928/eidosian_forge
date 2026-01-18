from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
from googlecloudsdk.core.util import tokenizer
import six
def _DictToOrderedDict(obj):
    """Recursively converts a JSON-serializable dict to an OrderedDict."""
    if isinstance(obj, dict):
        new_obj = collections.OrderedDict(sorted(obj.items()))
        for key, value in six.iteritems(new_obj):
            new_obj[key] = _DictToOrderedDict(value)
        return new_obj
    elif isinstance(obj, list):
        return [_DictToOrderedDict(item) for item in obj]
    else:
        return copy.deepcopy(obj)