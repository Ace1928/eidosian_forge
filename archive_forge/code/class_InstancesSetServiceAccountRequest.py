from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancesSetServiceAccountRequest(_messages.Message):
    """A InstancesSetServiceAccountRequest object.

  Fields:
    email: Email address of the service account.
    scopes: The list of scopes to be made available for this service account.
  """
    email = _messages.StringField(1)
    scopes = _messages.StringField(2, repeated=True)