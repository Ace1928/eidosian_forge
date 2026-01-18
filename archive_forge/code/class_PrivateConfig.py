from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateConfig(_messages.Message):
    """PrivateConfig includes settings for private instance.

  Fields:
    caPool: Required. Immutable. CA pool resource, resource must in the format
      of `projects/{project}/locations/{location}/caPools/{ca_pool}`.
    httpServiceAttachment: Output only. Service Attachment for HTTP, resource
      is in the format of `projects/{project}/regions/{region}/serviceAttachme
      nts/{service_attachment}`.
    isPrivate: Required. Immutable. Indicate if it's private instance.
    sshServiceAttachment: Output only. Service Attachment for SSH, resource is
      in the format of `projects/{project}/regions/{region}/serviceAttachments
      /{service_attachment}`.
  """
    caPool = _messages.StringField(1)
    httpServiceAttachment = _messages.StringField(2)
    isPrivate = _messages.BooleanField(3)
    sshServiceAttachment = _messages.StringField(4)