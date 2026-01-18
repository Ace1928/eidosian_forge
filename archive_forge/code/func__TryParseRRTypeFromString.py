from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from dns import rdatatype
from googlecloudsdk.api_lib.dns import import_util
from googlecloudsdk.api_lib.dns import record_types
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def _TryParseRRTypeFromString(type_str):
    """Tries to parse the rrtype wire value from the given string.

  Args:
    type_str: The record type as a string (e.g. "A", "MX"...).

  Raises:
    UnsupportedRecordType: If given record-set type is not supported

  Returns:
    The wire value rrtype as an int or rdatatype enum.
  """
    rd_type = rdatatype.from_text(type_str)
    if rd_type not in record_types.SUPPORTED_TYPES:
        raise UnsupportedRecordType('Unsupported record-set type [%s]' % type_str)
    return rd_type