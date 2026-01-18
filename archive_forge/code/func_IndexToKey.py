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
def IndexToKey(index):
    """Convert Index to key.

  Args:
    index: A datastore_index.Index instance (not None!).

  Returns:
    A tuple of the form (kind, ancestor, properties) where properties
    is a sequence of PropertySpec objects derived from the Index.
  """
    props = []
    if index.properties is not None:
        for prop in index.properties:
            props.append(PropertySpec(name=prop.name, direction=ASCENDING if prop.IsAscending() else DESCENDING))
    return (index.kind, index.ancestor, tuple(props))