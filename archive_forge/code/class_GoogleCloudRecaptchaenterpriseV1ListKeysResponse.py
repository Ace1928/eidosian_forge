from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1ListKeysResponse(_messages.Message):
    """Response to request to list keys in a project.

  Fields:
    keys: Key details.
    nextPageToken: Token to retrieve the next page of results. It is set to
      empty if no keys remain in results.
  """
    keys = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1Key', 1, repeated=True)
    nextPageToken = _messages.StringField(2)