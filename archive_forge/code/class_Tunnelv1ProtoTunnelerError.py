from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Tunnelv1ProtoTunnelerError(_messages.Message):
    """TunnelerError is an error proto for errors returned by the connection
  manager.

  Fields:
    err: Original raw error
    retryable: retryable isn't used for now, but we may want to reuse it in
      the future.
  """
    err = _messages.StringField(1)
    retryable = _messages.BooleanField(2)