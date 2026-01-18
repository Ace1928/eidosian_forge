from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MoveProjectRequest(_messages.Message):
    """The request sent to MoveProject method.

  Fields:
    destinationParent: Required. The new parent to move the Project under.
  """
    destinationParent = _messages.StringField(1)