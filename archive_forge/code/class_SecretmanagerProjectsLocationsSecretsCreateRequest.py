from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretmanagerProjectsLocationsSecretsCreateRequest(_messages.Message):
    """A SecretmanagerProjectsLocationsSecretsCreateRequest object.

  Fields:
    parent: Required. The resource name of the project to associate with the
      Secret, in the format `projects/*` or `projects/*/locations/*`.
    secret: A Secret resource to be passed as the request body.
    secretId: Required. This must be unique within the project. A secret ID is
      a string with a maximum length of 255 characters and can contain
      uppercase and lowercase letters, numerals, and the hyphen (`-`) and
      underscore (`_`) characters.
  """
    parent = _messages.StringField(1, required=True)
    secret = _messages.MessageField('Secret', 2)
    secretId = _messages.StringField(3)