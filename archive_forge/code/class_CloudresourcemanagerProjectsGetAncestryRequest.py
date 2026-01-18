from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerProjectsGetAncestryRequest(_messages.Message):
    """A CloudresourcemanagerProjectsGetAncestryRequest object.

  Fields:
    getAncestryRequest: A GetAncestryRequest resource to be passed as the
      request body.
    projectId: Required. The Project ID (for example, `my-project-123`).
  """
    getAncestryRequest = _messages.MessageField('GetAncestryRequest', 1)
    projectId = _messages.StringField(2, required=True)