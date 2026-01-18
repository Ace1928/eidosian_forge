from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServiceCitation(_messages.Message):
    """Source attributions for content.

  Fields:
    endIndex: End index into the content.
    license: License of the attribution.
    publicationDate: Publication date of the attribution.
    startIndex: Start index into the content.
    title: Title of the attribution.
    uri: Url reference of the attribution.
  """
    endIndex = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    license = _messages.StringField(2)
    publicationDate = _messages.MessageField('GoogleTypeDate', 3)
    startIndex = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    title = _messages.StringField(5)
    uri = _messages.StringField(6)