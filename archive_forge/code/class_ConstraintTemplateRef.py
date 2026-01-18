from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConstraintTemplateRef(_messages.Message):
    """ConstraintTemplateRef identifies a constraint template.

  Fields:
    name: The constraint template name.
  """
    name = _messages.StringField(1)