from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PscConfig(_messages.Message):
    """Information for Private Service Connect (PSC) setup for a Looker
  instance.

  Fields:
    allowedVpcs: Optional. List of VPCs that are allowed ingress into looker.
      Format: projects/{project}/global/networks/{network}
    lookerServiceAttachmentUri: Output only. URI of the Looker service
      attachment.
    serviceAttachments: Optional. List of egress service attachment
      configurations.
  """
    allowedVpcs = _messages.StringField(1, repeated=True)
    lookerServiceAttachmentUri = _messages.StringField(2)
    serviceAttachments = _messages.MessageField('ServiceAttachment', 3, repeated=True)