from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventFilter(_messages.Message):
    """Filters events based on exact matches on the CloudEvents attributes.

  Fields:
    attribute: Required. The name of a CloudEvents attribute. Currently, only
      a subset of attributes are supported for filtering. You can [retrieve a
      specific provider's supported event types](/eventarc/docs/list-
      providers#describe-provider). All triggers MUST provide a filter for the
      'type' attribute.
    operator: Optional. The operator used for matching the events with the
      value of the filter. If not specified, only events that have an exact
      key-value pair specified in the filter are matched. The allowed values
      are `path_pattern` and `match-path-pattern`. `path_pattern` is only
      allowed for GCFv1 triggers.
    value: Required. The value for the attribute.
  """
    attribute = _messages.StringField(1)
    operator = _messages.StringField(2)
    value = _messages.StringField(3)