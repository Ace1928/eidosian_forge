from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MatchingCriteria(_messages.Message):
    """Matches events based on exact matches on the CloudEvents attributes.

  Fields:
    attribute: Required. The name of a CloudEvents attribute. Currently, only
      a subset of attributes can be specified. All triggers MUST provide a
      matching criteria for the 'type' attribute.
    value: Required. The value for the attribute.
  """
    attribute = _messages.StringField(1)
    value = _messages.StringField(2)