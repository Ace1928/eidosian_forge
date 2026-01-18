from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootMetricOutput(_messages.Message):
    """A LearningGenaiRootMetricOutput object.

  Fields:
    debug: A string attribute.
    name: Name of the metric.
    numericValue: A number attribute.
    status: A UtilStatusProto attribute.
    stringValue: A string attribute.
  """
    debug = _messages.StringField(1)
    name = _messages.StringField(2)
    numericValue = _messages.FloatField(3)
    status = _messages.MessageField('UtilStatusProto', 4)
    stringValue = _messages.StringField(5)