from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Egress(_messages.Message):
    """The details of the egress info. One of the following options should be
  set.

  Fields:
    peeredVpc: A VPC from the consumer project.
  """
    peeredVpc = _messages.MessageField('PeeredVpc', 1)