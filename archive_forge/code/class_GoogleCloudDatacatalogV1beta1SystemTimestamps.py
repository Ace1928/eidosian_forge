from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1SystemTimestamps(_messages.Message):
    """Timestamps about this resource according to a particular system.

  Fields:
    createTime: The creation time of the resource within the given system.
    expireTime: Output only. The expiration time of the resource within the
      given system. Currently only apllicable to BigQuery resources.
    updateTime: The last-modified time of the resource within the given
      system.
  """
    createTime = _messages.StringField(1)
    expireTime = _messages.StringField(2)
    updateTime = _messages.StringField(3)