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
def GetRecommendedIndexProperties(properties):
    """Converts the properties returned by datastore_index.CompositeIndexForQuery
  into a recommended list of index properties with the desired constraints.

  Sets (of property names or PropertySpec objects) are sorted, so as to
  normalize them.

  Args:
    properties: See datastore_index.CompositeIndexForQuery

  Returns:
    A tuple of PropertySpec objects.

  """
    prefix, postfix = properties
    result = []
    for sub_list in itertools.chain((prefix,), postfix):
        if isinstance(sub_list, (frozenset, set)):
            result.extend([p if isinstance(p, PropertySpec) else PropertySpec(name=p) for p in sorted(sub_list)])
        else:
            result.extend([PropertySpec(name=p.name, direction=ASCENDING) if p.direction is None else p for p in sub_list])
    return tuple(result)