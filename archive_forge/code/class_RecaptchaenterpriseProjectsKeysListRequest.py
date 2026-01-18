from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsKeysListRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsKeysListRequest object.

  Fields:
    pageSize: Optional. The maximum number of keys to return. Default is 10.
      Max limit is 1000.
    pageToken: Optional. The next_page_token value returned from a previous.
      ListKeysRequest, if any.
    parent: Required. The name of the project that contains the keys that will
      be listed, in the format `projects/{project}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)