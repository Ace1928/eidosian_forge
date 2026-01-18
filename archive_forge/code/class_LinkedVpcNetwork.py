from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkedVpcNetwork(_messages.Message):
    """An existing VPC network.

  Fields:
    excludeExportRanges: Optional. IP ranges encompassing the subnets to be
      excluded from peering.
    includeExportRanges: Optional. IP ranges allowed to be included from
      peering.
    uri: Required. The URI of the VPC network resource.
  """
    excludeExportRanges = _messages.StringField(1, repeated=True)
    includeExportRanges = _messages.StringField(2, repeated=True)
    uri = _messages.StringField(3)