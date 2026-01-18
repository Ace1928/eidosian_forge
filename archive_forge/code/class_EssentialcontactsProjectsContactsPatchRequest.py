from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EssentialcontactsProjectsContactsPatchRequest(_messages.Message):
    """A EssentialcontactsProjectsContactsPatchRequest object.

  Fields:
    googleCloudEssentialcontactsV1Contact: A
      GoogleCloudEssentialcontactsV1Contact resource to be passed as the
      request body.
    name: The identifier for the contact. Format:
      {resource_type}/{resource_id}/contacts/{contact_id}
    updateMask: Optional. The update mask applied to the resource. For the
      `FieldMask` definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    googleCloudEssentialcontactsV1Contact = _messages.MessageField('GoogleCloudEssentialcontactsV1Contact', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)