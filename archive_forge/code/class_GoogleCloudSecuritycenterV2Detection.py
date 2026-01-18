from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Detection(_messages.Message):
    """Memory hash detection contributing to the binary family match.

  Fields:
    binary: The name of the binary associated with the memory hash signature
      detection.
    percentPagesMatched: The percentage of memory page hashes in the signature
      that were matched.
  """
    binary = _messages.StringField(1)
    percentPagesMatched = _messages.FloatField(2)