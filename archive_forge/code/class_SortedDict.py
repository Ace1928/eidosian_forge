from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class SortedDict(dict):
    """Represents a dict with a particular key order for yaml representing."""

    def __init__(self, keys, data):
        super(SortedDict, self).__init__()
        self.keys = keys
        self.update(data)

    def ordered_items(self):
        result = []
        for key in self.keys:
            if self.get(key) is not None:
                result.append((key, self.get(key)))
        return result