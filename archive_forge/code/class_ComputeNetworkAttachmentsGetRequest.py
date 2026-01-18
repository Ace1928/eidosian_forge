from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeNetworkAttachmentsGetRequest(_messages.Message):
    """A ComputeNetworkAttachmentsGetRequest object.

  Fields:
    networkAttachment: Name of the NetworkAttachment resource to return.
    project: Project ID for this request.
    region: Name of the region of this request.
  """
    networkAttachment = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)