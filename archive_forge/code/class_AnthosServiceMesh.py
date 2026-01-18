from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnthosServiceMesh(_messages.Message):
    """Message describing AnthosServiceMesh based workload object.

  Fields:
    serviceAccount: Immutable. workload ID = IAM Service account
  """
    serviceAccount = _messages.StringField(1)