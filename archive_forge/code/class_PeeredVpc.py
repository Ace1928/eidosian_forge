from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PeeredVpc(_messages.Message):
    """The peered VPC owned by the consumer project.

  Fields:
    networkVpc: Required. The name of the peered VPC owned by the consumer
      project.
  """
    networkVpc = _messages.StringField(1)