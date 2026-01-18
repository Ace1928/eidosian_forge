from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchGetServicesResponse(_messages.Message):
    """Response message for the `BatchGetServices` method.

  Fields:
    services: The requested Service states.
  """
    services = _messages.MessageField('GoogleApiServiceusageV1Service', 1, repeated=True)