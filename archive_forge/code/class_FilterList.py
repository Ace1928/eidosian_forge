from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FilterList(_messages.Message):
    """List of infoTypes to be filtered.

  Fields:
    infoTypes: These infoTypes are based on after the `eval_info_type_mapping`
      and `golden_info_type_mapping`.
  """
    infoTypes = _messages.StringField(1, repeated=True)