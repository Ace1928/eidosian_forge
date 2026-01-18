from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetSslProxiesSetBackendServiceRequest(_messages.Message):
    """A TargetSslProxiesSetBackendServiceRequest object.

  Fields:
    service: The URL of the new BackendService resource for the
      targetSslProxy.
  """
    service = _messages.StringField(1)