from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StringList(_messages.Message):
    """A metric value representing a list of strings.

  Fields:
    elements: Elements of the list.
  """
    elements = _messages.StringField(1, repeated=True)