from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UrlList(_messages.Message):
    """UrlList proto helps users to set reusable, independently manageable
  lists of hosts, host patterns, URLs, URL patterns.

  Fields:
    createTime: Output only. Time when the security policy was created.
    description: Optional. Free-text description of the resource.
    name: Required. Name of the resource provided by the user. Name is of the
      form projects/{project}/locations/{location}/urlLists/{url_list}
      url_list should match the pattern:(^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$).
    updateTime: Output only. Time when the security policy was updated.
    values: Required. FQDNs and URLs.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    name = _messages.StringField(3)
    updateTime = _messages.StringField(4)
    values = _messages.StringField(5, repeated=True)