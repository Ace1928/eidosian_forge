from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsRepairRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsManagementDnsZoneBindingsRep
  airRequest object.

  Fields:
    name: Required. The resource name of the management DNS zone binding to
      repair. Resource names are schemeless URIs that follow the conventions
      in https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud/managementDnsZoneBindings/my-management-dns-zone-binding`
    repairManagementDnsZoneBindingRequest: A
      RepairManagementDnsZoneBindingRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    repairManagementDnsZoneBindingRequest = _messages.MessageField('RepairManagementDnsZoneBindingRequest', 2)