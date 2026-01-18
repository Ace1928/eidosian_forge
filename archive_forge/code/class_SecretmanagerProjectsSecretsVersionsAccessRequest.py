from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretmanagerProjectsSecretsVersionsAccessRequest(_messages.Message):
    """A SecretmanagerProjectsSecretsVersionsAccessRequest object.

  Fields:
    name: Required. The resource name of the SecretVersion in the format
      `projects/*/secrets/*/versions/*` or
      `projects/*/locations/*/secrets/*/versions/*`.
      `projects/*/secrets/*/versions/latest` or
      `projects/*/locations/*/secrets/*/versions/latest` is an alias to the
      most recently created SecretVersion.
  """
    name = _messages.StringField(1, required=True)