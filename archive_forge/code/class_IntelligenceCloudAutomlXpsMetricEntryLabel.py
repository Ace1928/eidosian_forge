from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntelligenceCloudAutomlXpsMetricEntryLabel(_messages.Message):
    """A IntelligenceCloudAutomlXpsMetricEntryLabel object.

  Fields:
    labelName: The name of the label.
    labelValue: The value of the label.
  """
    labelName = _messages.StringField(1)
    labelValue = _messages.StringField(2)