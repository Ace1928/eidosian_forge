from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaPartnerTenant(_messages.Message):
    """Information about a BeyoncCorp Enterprise PartnerTenant.

  Fields:
    createTime: Output only. Timestamp when the resource was created.
    displayName: Optional. An arbitrary caller-provided name for the
      PartnerTenant. Cannot exceed 64 characters.
    group: Optional. Group information for the users enabled to use the
      partnerTenant. If the group information is not provided then the
      partnerTenant will be enabled for all users.
    name: Output only. Unique resource name of the PartnerTenant. The name is
      ignored when creating PartnerTenant.
    partnerMetadata: Optional. Metadata provided by the Partner associated
      with PartnerTenant.
    updateTime: Output only. Timestamp when the resource was last modified.
  """
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    group = _messages.MessageField('GoogleCloudBeyondcorpPartnerservicesV1alphaGroup', 3)
    name = _messages.StringField(4)
    partnerMetadata = _messages.MessageField('GoogleCloudBeyondcorpPartnerservicesV1alphaPartnerMetadata', 5)
    updateTime = _messages.StringField(6)