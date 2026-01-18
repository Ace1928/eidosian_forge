from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysProjectsKeysGetRequest(_messages.Message):
    """A ApikeysProjectsKeysGetRequest object.

  Fields:
    name: Required. The resource name of the API key to be retrieved.
  """
    name = _messages.StringField(1, required=True)