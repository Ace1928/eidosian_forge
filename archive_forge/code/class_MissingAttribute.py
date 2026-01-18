from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class MissingAttribute(ValidationError):
    """Raised when a required attribute is missing from object."""

    def __init__(self, key):
        msg = 'Missing required value [{}].'.format(key)
        super(MissingAttribute, self).__init__(msg)