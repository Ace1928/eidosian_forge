from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Citation(_messages.Message):
    """Source attributions for content.

  Fields:
    endIndex: Output only. End index into the content.
    license: Output only. License of the attribution.
    publicationDate: Output only. Publication date of the attribution.
    startIndex: Output only. Start index into the content.
    title: Output only. Title of the attribution.
    uri: Output only. Url reference of the attribution.
  """
    endIndex = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    license = _messages.StringField(2)
    publicationDate = _messages.MessageField('GoogleTypeDate', 3)
    startIndex = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    title = _messages.StringField(5)
    uri = _messages.StringField(6)