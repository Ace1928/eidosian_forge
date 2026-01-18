from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FacetValue(_messages.Message):
    """FacetValue represents a single value result in a facet.

  Fields:
    count: Output only. Number of videos or segments corresponding to the
      facet.
    stringValue: A string attribute.
  """
    count = _messages.IntegerField(1)
    stringValue = _messages.StringField(2)