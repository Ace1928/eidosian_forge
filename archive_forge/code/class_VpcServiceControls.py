from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcServiceControls(_messages.Message):
    """Response for the get VPC Service Controls request.

  Fields:
    enabled: Output only. Indicates whether the VPC Service Controls are
      enabled or disabled for the connection. If the consumer called the
      EnableVpcServiceControls method, then this is true. If the consumer
      called DisableVpcServiceControls, then this is false. The default is
      false.
  """
    enabled = _messages.BooleanField(1)