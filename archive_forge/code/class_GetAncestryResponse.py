from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetAncestryResponse(_messages.Message):
    """Response from the projects.getAncestry method.

  Fields:
    ancestor: Ancestors are ordered from bottom to top of the resource
      hierarchy. The first ancestor is the project itself, followed by the
      project's parent, etc..
  """
    ancestor = _messages.MessageField('Ancestor', 1, repeated=True)