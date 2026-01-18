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
def _NullTranslation(rdata, origin=None):
    """Returns the given rdata as text (formatted by its .to_text() method).

  Args:
    rdata: Rdata, The data to be translated.
    origin: Name, The origin domain name.

  Returns:
    str, The textual presentation form of the given rdata.
  """
    return rdata.to_text(origin=origin, relativize=False)