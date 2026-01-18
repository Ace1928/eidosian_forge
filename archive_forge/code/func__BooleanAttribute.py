from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.api.validation import ValidationError
from googlecloudsdk.third_party.appengine.datastore.datastore_index import Index
from googlecloudsdk.third_party.appengine.datastore.datastore_index import IndexDefinitions
from googlecloudsdk.third_party.appengine.datastore.datastore_index import Property
def _BooleanAttribute(value):
    """Parse the given attribute value as a Boolean value.

  This follows the specification here:
  http://www.w3.org/TR/2012/REC-xmlschema11-2-20120405/datatypes.html#boolean

  Args:
    value: the value to parse.

  Returns:
    True if the value parses as true, False if it parses as false, None if it
    parses as neither.
  """
    if value in ['true', '1']:
        return True
    elif value in ['false', '0']:
        return False
    else:
        return None