from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CopyInstanceRequest(_messages.Message):
    """CopyInstanceRequest copies a fileshare from a Basic instance to an
  Enterprise or High Scale instance.

  Fields:
    reservedIpRange: The IP address range in the VPC of the source and target
      instances that can be used for networking resources that need to be
      created by the service for the subnet needed during the operation to
      perform the copy. If the instance uses IP-based access control, the
      source instance must allow reads from the IP range and the target
      instance must allow writes from the IP range. It must be a /29 CIDR
      block in the internal IP address ranges
      (https://www.arin.net/knowledge/address_filters.html). For example,
      10.0.0.0/29.
    sourceFileShare: Required. Name of the source file share in the Filestore
      Basic instance that we are copying the data from.
    sourceInstance: Required. The name of the Basic instance that we are
      copying the fileshare from, in the format `projects/{project_number}/loc
      ations/{location}/instances/{instance_id}`.
    targetFileShare: Required. Name of the target file share in the Filestore
      High Scale or Enterprise instance that we are copying the data to.
  """
    reservedIpRange = _messages.StringField(1)
    sourceFileShare = _messages.StringField(2)
    sourceInstance = _messages.StringField(3)
    targetFileShare = _messages.StringField(4)