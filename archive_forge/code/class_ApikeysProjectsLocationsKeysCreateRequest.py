from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysProjectsLocationsKeysCreateRequest(_messages.Message):
    """A ApikeysProjectsLocationsKeysCreateRequest object.

  Fields:
    keyId: User specified key id (optional). If specified, it will become the
      final component of the key resource name. The id must be unique within
      the project, must conform with RFC-1034, is restricted to lower-cased
      letters, and has a maximum length of 63 characters. In another word, the
      id must match the regular expression: `[a-z]([a-z0-9-]{0,61}[a-z0-9])?`.
      The id must NOT be a UUID-like string.
    parent: Required. The project in which the API key is created.
    v2Key: A V2Key resource to be passed as the request body.
  """
    keyId = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)
    v2Key = _messages.MessageField('V2Key', 3)