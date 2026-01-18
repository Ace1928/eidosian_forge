from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetupServiceResponse(_messages.Message):
    """Response message for `SetupService` method.

  Fields:
    serviceAccount: The service account that the service will use to act on
      resources.
  """
    serviceAccount = _messages.StringField(1)