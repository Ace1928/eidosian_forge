from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricStructuredName(_messages.Message):
    """Identifies a metric, by describing the source which generated the
  metric.

  Messages:
    ContextValue: Zero or more labeled fields which identify the part of the
      job this metric is associated with, such as the name of a step or
      collection. For example, built-in counters associated with steps will
      have context['step'] = . Counters associated with PCollections in the
      SDK will have context['pcollection'] = .

  Fields:
    context: Zero or more labeled fields which identify the part of the job
      this metric is associated with, such as the name of a step or
      collection. For example, built-in counters associated with steps will
      have context['step'] = . Counters associated with PCollections in the
      SDK will have context['pcollection'] = .
    name: Worker-defined metric name.
    origin: Origin (namespace) of metric name. May be blank for user-define
      metrics; will be "dataflow" for metrics defined by the Dataflow service
      or SDK.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ContextValue(_messages.Message):
        """Zero or more labeled fields which identify the part of the job this
    metric is associated with, such as the name of a step or collection. For
    example, built-in counters associated with steps will have context['step']
    = . Counters associated with PCollections in the SDK will have
    context['pcollection'] = .

    Messages:
      AdditionalProperty: An additional property for a ContextValue object.

    Fields:
      additionalProperties: Additional properties of type ContextValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ContextValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    context = _messages.MessageField('ContextValue', 1)
    name = _messages.StringField(2)
    origin = _messages.StringField(3)