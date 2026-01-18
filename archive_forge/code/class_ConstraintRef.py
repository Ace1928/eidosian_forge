from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConstraintRef(_messages.Message):
    """Constraint represents a single constraint. Base identifying resource.

  Fields:
    constraintTemplateName: The constraint template name, lowercase of the
      constraint kind. Used for identification, not for UI display.
    name: The constraint name.
  """
    constraintTemplateName = _messages.StringField(1)
    name = _messages.StringField(2)