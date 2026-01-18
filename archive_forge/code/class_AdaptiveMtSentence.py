from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdaptiveMtSentence(_messages.Message):
    """An AdaptiveMt sentence entry.

  Fields:
    createTime: Output only. Timestamp when this sentence was created.
    name: Required. The resource name of the file, in form of
      `projects/{project-number-or-id}/locations/{location_id}/adaptiveMtDatas
      ets/{dataset}/adaptiveMtFiles/{file}/adaptiveMtSentences/{sentence}`
    sourceSentence: Required. The source sentence.
    targetSentence: Required. The target sentence.
    updateTime: Output only. Timestamp when this sentence was last updated.
  """
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)
    sourceSentence = _messages.StringField(3)
    targetSentence = _messages.StringField(4)
    updateTime = _messages.StringField(5)