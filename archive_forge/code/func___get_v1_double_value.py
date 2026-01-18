from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __get_v1_double_value(self, v1_value):
    """Returns a double value from a v1 Value.

    Args:
      v1_value: an googledatastore.Value

    Returns:
      a double

    Raises:
      InvalidConversionError: if the value doesn't contain a double value
    """
    check_conversion(v1_value.HasField('double_value'), 'Value does not contain a double value.')
    return v1_value.double_value