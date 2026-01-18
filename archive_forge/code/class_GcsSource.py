from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcsSource(_messages.Message):
    """The Google Cloud Storage location for the input content.

  Fields:
    inputUri: Required. Source data URI. For example,
      `gs://my_bucket/my_object`.
  """
    inputUri = _messages.StringField(1)