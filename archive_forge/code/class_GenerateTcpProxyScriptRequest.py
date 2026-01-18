from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateTcpProxyScriptRequest(_messages.Message):
    """Request message for 'GenerateTcpProxyScript' request.

  Fields:
    vmMachineType: Required. The type of the Compute instance that will host
      the proxy.
    vmName: Required. The name of the Compute instance that will host the
      proxy.
    vmSubnet: Required. The name of the subnet the Compute instance will use
      for private connectivity. Must be supplied in the form of
      projects/{project}/regions/{region}/subnetworks/{subnetwork}. Note: the
      region for the subnet must match the Compute instance region.
    vmZone: Optional. The Google Cloud Platform zone to create the VM in. The
      fully qualified name of the zone must be specified, including the region
      name, for example "us-central1-b". If not specified, uses the "-b" zone
      of the destination Connection Profile's region.
  """
    vmMachineType = _messages.StringField(1)
    vmName = _messages.StringField(2)
    vmSubnet = _messages.StringField(3)
    vmZone = _messages.StringField(4)