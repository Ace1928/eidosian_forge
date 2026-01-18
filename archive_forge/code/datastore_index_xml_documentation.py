from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.api.validation import ValidationError
from googlecloudsdk.third_party.appengine.datastore.datastore_index import Index
from googlecloudsdk.third_party.appengine.datastore.datastore_index import IndexDefinitions
from googlecloudsdk.third_party.appengine.datastore.datastore_index import Property
Processes XML <datastore-index> nodes into Index objects.

    The following information is parsed out:
    kind: specifies the kind of entities to index.
    ancestor: true if the index supports queries that filter by
      ancestor-key to constraint results to a single entity group.
    property: represents the entity properties to index, with a name
      and direction attribute.

    Args:
      node: <datastore-index> XML node in datastore-indexes.xml.
    