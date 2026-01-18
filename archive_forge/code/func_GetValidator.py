from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
@classmethod
def GetValidator(cls, key):
    """Check the key for validity and return a corresponding value validator.

    Args:
      key: The key that will correspond to the validator we are returning.
    """
    key = AsValidator(cls.KEY_VALIDATOR)(key, 'key in %s' % cls.__name__)
    return AsValidator(cls.VALUE_VALIDATOR)