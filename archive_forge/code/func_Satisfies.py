from __future__ import absolute_import
from ruamel import yaml
import copy
import itertools
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_object
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
def Satisfies(self, other):
    """Determines whether existing index can satisfy requirements of a new query.

    Used in finding matching postfix with traditional "ordered" index specs.
    """
    assert isinstance(other, PropertySpec)
    if self._name != other._name:
        return False
    if self._mode is not None or other._mode is not None:
        return False
    if other._direction is None:
        return True
    return self._direction == other._direction