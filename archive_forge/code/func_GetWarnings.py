from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
def GetWarnings(self, value, key, obj):
    del obj
    if value is not None:
        return [(key, 'Field %s is deprecated; use %s instead.' % (key, self.preferred))]
    return []