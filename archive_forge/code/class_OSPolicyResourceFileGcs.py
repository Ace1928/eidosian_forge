from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceFileGcs(_messages.Message):
    """Specifies a file available as a Cloud Storage Object.

  Fields:
    bucket: Required. Bucket of the Cloud Storage object.
    generation: Generation number of the Cloud Storage object.
    object: Required. Name of the Cloud Storage object.
  """
    bucket = _messages.StringField(1)
    generation = _messages.IntegerField(2)
    object = _messages.StringField(3)