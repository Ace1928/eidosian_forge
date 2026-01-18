from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PropertyReference(_messages.Message):
    """A reference to a property relative to the kind expressions.

  Fields:
    name: A reference to a property. Requires: * MUST be a dot-delimited (`.`)
      string of segments, where each segment conforms to entity property name
      limitations.
  """
    name = _messages.StringField(1)