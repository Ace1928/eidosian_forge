from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotifierSecret(_messages.Message):
    """NotifierSecret is the container that maps a secret name (reference) to
  its Google Cloud Secret Manager resource path.

  Fields:
    name: Name is the local name of the secret, such as the verbatim string
      "my-smtp-password".
    value: Value is interpreted to be a resource path for fetching the actual
      (versioned) secret data for this secret. For example, this would be a
      Google Cloud Secret Manager secret version resource path like:
      "projects/my-project/secrets/my-secret/versions/latest".
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)