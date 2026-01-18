from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FieldReference(_messages.Message):
    """A reference to a field in a document, ex: `stats.operations`.

  Fields:
    fieldPath: A reference to a field in a document. Requires: * MUST be a
      dot-delimited (`.`) string of segments, where each segment conforms to
      document field name limitations.
  """
    fieldPath = _messages.StringField(1)