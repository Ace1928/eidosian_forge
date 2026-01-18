from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootRoutingDecisionMetadataTokenLengthBasedModelInputTokenMetadata(_messages.Message):
    """A LearningGenaiRootRoutingDecisionMetadataTokenLengthBasedModelInputToke
  nMetadata object.

  Fields:
    computedInputTokenLength: The length computed by backends using the
      formatter & tokenizer specific to the model
    modelId: A string attribute.
    pickedAsFallback: If true, the model was selected as a fallback, since no
      model met requirements.
    selected: If true, the model was selected since it met the requriements.
  """
    computedInputTokenLength = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    modelId = _messages.StringField(2)
    pickedAsFallback = _messages.BooleanField(3)
    selected = _messages.BooleanField(4)