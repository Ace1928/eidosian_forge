from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding as api_encoding
from dns import rdatatype
from dns import zone
from googlecloudsdk.api_lib.dns import record_types
from googlecloudsdk.api_lib.dns import svcb_stub
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
import six
def _ToStandardEnumTypeSafe(string_type):
    """Converts string_type to an RdataType enum value if it is a standard type.

  Only standard record types can be converted to a RdataType, all other types
  will cause an exception. This method allow getting the standard enum type if
  available without throwing an exception if an extended type is provided.

  Args:
    string_type: [str] The record type as a string.

  Returns:
    The record type as an RdataType enum or None if the type is not a standard
    DNS type.
  """
    if string_type in record_types.CLOUD_DNS_EXTENDED_TYPES:
        return None
    return rdatatype.from_text(string_type)