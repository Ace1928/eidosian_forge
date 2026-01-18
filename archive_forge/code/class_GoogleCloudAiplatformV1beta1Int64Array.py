from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Int64Array(_messages.Message):
    """A list of int64 values.

  Fields:
    values: A list of int64 values.
  """
    values = _messages.IntegerField(1, repeated=True)