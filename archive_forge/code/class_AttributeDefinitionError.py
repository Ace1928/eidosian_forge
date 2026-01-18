from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class AttributeDefinitionError(Error):
    """An error occurred in the definition of class attributes."""