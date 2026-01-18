from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.api.validation import ValidationError
from googlecloudsdk.third_party.appengine.datastore.datastore_index import Index
from googlecloudsdk.third_party.appengine.datastore.datastore_index import IndexDefinitions
from googlecloudsdk.third_party.appengine.datastore.datastore_index import Property
def IndexesXmlToIndexDefinitions(xml_str):
    """Convert a <datastore-indexes> XML string into an IndexDefinitions objects.

  Args:
    xml_str: a string containing a complete XML document where the root node is
      <datastore-indexes>.

  Returns:
    an IndexDefinitions object parsed out of the XML string.

  Raises:
    ValidationError: in case of malformed XML or illegal inputs.
  """
    parser = IndexesXmlParser()
    return parser.Parse(xml_str)