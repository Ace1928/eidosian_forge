from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdaptiveMtDataset(_messages.Message):
    """An Adaptive MT Dataset.

  Fields:
    createTime: Output only. Timestamp when this dataset was created.
    displayName: The name of the dataset to show in the interface. The name
      can be up to 32 characters long and can consist only of ASCII Latin
      letters A-Z and a-z, underscores (_), and ASCII digits 0-9.
    exampleCount: The number of examples in the dataset.
    name: Required. The resource name of the dataset, in form of
      `projects/{project-number-or-
      id}/locations/{location_id}/adaptiveMtDatasets/{dataset_id}`
    sourceLanguageCode: The BCP-47 language code of the source language.
    targetLanguageCode: The BCP-47 language code of the target language.
    updateTime: Output only. Timestamp when this dataset was last updated.
  """
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    exampleCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    name = _messages.StringField(4)
    sourceLanguageCode = _messages.StringField(5)
    targetLanguageCode = _messages.StringField(6)
    updateTime = _messages.StringField(7)