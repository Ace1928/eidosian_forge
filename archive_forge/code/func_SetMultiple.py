from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
def SetMultiple(self, attributes):
    """Set multiple values on Validated instance.

    All attributes will be validated before being set.

    Args:
      attributes: A dict of attributes/items to set.

    Raises:
      ValidationError: when no validated attribute exists on class.
    """
    for key, value in attributes.items():
        self.Set(key, value)