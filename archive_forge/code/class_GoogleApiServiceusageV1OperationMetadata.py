from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceusageV1OperationMetadata(_messages.Message):
    """The operation metadata returned for the batchend services operation.

  Fields:
    resourceNames: The full name of the resources that this operation is
      directly associated with.
  """
    resourceNames = _messages.StringField(1, repeated=True)