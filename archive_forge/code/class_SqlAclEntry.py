from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlAclEntry(_messages.Message):
    """An entry for an Access Control list.

  Fields:
    expireTime: The time when this access control entry expires in [RFC
      3339](https://tools.ietf.org/html/rfc3339) format, for example:
      `2012-11-15T16:19:00.094Z`.
    label: A label to identify this entry.
    value: The allowlisted value for the access control list.
  """
    expireTime = _messages.StringField(1)
    label = _messages.StringField(2)
    value = _messages.StringField(3)