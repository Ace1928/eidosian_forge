from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReimageNodeRequest(_messages.Message):
    """Request for ReimageNode.

  Fields:
    tensorflowVersion: The version for reimage to create.
  """
    tensorflowVersion = _messages.StringField(1)