from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UndeleteServiceResponse(_messages.Message):
    """Response message for UndeleteService method.

  Fields:
    service: Revived service resource.
  """
    service = _messages.MessageField('ManagedService', 1)