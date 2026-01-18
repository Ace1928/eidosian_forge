from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IpRange(_messages.Message):
    """An IP range provided in any one of the supported formats.

  Fields:
    externalAddress: The name of an `ExternalAddress` resource. The external
      address must have been reserved in the scope of this external access
      rule's parent network policy. Provide the external address name in the
      form of `projects/{project}/locations/{location}/privateClouds/{private_
      cloud}/externalAddresses/{external_address}`. For example: `projects/my-
      project/locations/us-central1-a/privateClouds/my-
      cloud/externalAddresses/my-address`.
    ipAddress: A single IP address. For example: `10.0.0.5`.
    ipAddressRange: An IP address range in the CIDR format. For example:
      `10.0.0.0/24`.
  """
    externalAddress = _messages.StringField(1)
    ipAddress = _messages.StringField(2)
    ipAddressRange = _messages.StringField(3)