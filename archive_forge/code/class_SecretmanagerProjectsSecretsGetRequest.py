from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretmanagerProjectsSecretsGetRequest(_messages.Message):
    """A SecretmanagerProjectsSecretsGetRequest object.

  Fields:
    name: Required. The resource name of the Secret, in the format
      `projects/*/secrets/*` or `projects/*/locations/*/secrets/*`.
  """
    name = _messages.StringField(1, required=True)