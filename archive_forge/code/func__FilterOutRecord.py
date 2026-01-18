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
def _FilterOutRecord(name, rdtype, origin, replace_origin_ns=False):
    """Returns whether the given record should be filtered out.

  Args:
    name: string, The name of the resord set we are considering
    rdtype: RDataType or string, type of Record we are considering approving.
    origin: Name, The origin domain of the zone we are considering
    replace_origin_ns: Bool, Whether origin NS records should be imported

  Returns:
    True if the given record should be filtered out, false otherwise.
  """
    if replace_origin_ns:
        return False
    elif name == origin and rdtype == rdatatype.NS:
        return True
    else:
        return False