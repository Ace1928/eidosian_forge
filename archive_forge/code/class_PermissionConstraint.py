from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PermissionConstraint(_messages.Message):
    """Custom constraint that specifies a key and a list of allowed values for
  Istio attributes.

  Fields:
    key: Key of the constraint.
    values: A list of allowed values.
  """
    key = _messages.StringField(1)
    values = _messages.StringField(2, repeated=True)