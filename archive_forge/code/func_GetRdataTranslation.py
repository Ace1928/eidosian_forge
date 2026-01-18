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
def GetRdataTranslation(rr_type):
    """Returns the rdata translation function for a record type.

  Args:
    rr_type: The record type

  Returns:
    The record type's translation function.
  """
    if rr_type == rdatatype.SOA:
        return _SOATranslation
    return _NullTranslation