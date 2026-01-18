from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GroundingAttributionWeb(_messages.Message):
    """Attribution from the web.

  Fields:
    title: Output only. Title of the attribution.
    uri: Output only. URI reference of the attribution.
  """
    title = _messages.StringField(1)
    uri = _messages.StringField(2)