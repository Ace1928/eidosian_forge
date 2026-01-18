from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CategoricalValue(_messages.Message):
    """Representative value of a categorical feature.

  Fields:
    categoryCounts: Counts of all categories for the categorical feature. If
      there are more than ten categories, we return top ten (by count) and
      return one more CategoryCount with category "_OTHER_" and count as
      aggregate counts of remaining categories.
  """
    categoryCounts = _messages.MessageField('CategoryCount', 1, repeated=True)