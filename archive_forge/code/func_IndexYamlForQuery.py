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
def IndexYamlForQuery(kind, ancestor, props):
    """Return the composite index definition YAML needed for a query.

  Given a query, the arguments for this method can be computed with:
    _, kind, ancestor, props = datastore_index.CompositeIndexForQuery(query)
    props = datastore_index.GetRecommendedIndexProperties(props)

  Args:
    kind: the kind or None
    ancestor: True if this is an ancestor query, False otherwise
    props: PropertySpec objects

  Returns:
    A string with the YAML for the composite index needed by the query.
  """
    serialized_yaml = []
    serialized_yaml.append('- kind: %s' % kind)
    if ancestor:
        serialized_yaml.append('  ancestor: yes')
    if props:
        serialized_yaml.append('  properties:')
        for prop in props:
            serialized_yaml.append('  - name: %s' % prop.name)
            if prop.direction == DESCENDING:
                serialized_yaml.append('    direction: desc')
            if prop.mode is GEOSPATIAL:
                serialized_yaml.append('    mode: geospatial')
    return '\n'.join(serialized_yaml)