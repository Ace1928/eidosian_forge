from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StringRestrictions(_messages.Message):
    """Restrictions on STRING type values

  Fields:
    allowedValues: The list of allowed values, if bounded. This field will be
      empty if there is a unbounded number of allowed values.
  """
    allowedValues = _messages.StringField(1, repeated=True)