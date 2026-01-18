from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StringArray(_messages.Message):
    """A list of string values.

  Fields:
    values: A list of string values.
  """
    values = _messages.StringField(1, repeated=True)