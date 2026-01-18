from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceGroup(_messages.Message):
    """Resource groups provide a mechanism to group OS policy resources.
  Resource groups enable OS policy authors to create a single OS policy to be
  applied to VMs running different operating Systems. When the OS policy is
  applied to a target VM, the appropriate resource group within the OS policy
  is selected based on the `OSFilter` specified within the resource group.

  Fields:
    inventoryFilters: List of inventory filters for the resource group. The
      resources in this resource group are applied to the target VM if it
      satisfies at least one of the following inventory filters. For example,
      to apply this resource group to VMs running either `RHEL` or `CentOS`
      operating systems, specify 2 items for the list with following values:
      inventory_filters[0].os_short_name='rhel' and
      inventory_filters[1].os_short_name='centos' If the list is empty, this
      resource group will be applied to the target VM unconditionally.
    osFilter: Deprecated. Use the `inventory_filters` field instead. Used to
      specify the OS filter for a resource group
    resources: Required. List of resources configured for this resource group.
      The resources are executed in the exact order specified here.
  """
    inventoryFilters = _messages.MessageField('OSPolicyInventoryFilter', 1, repeated=True)
    osFilter = _messages.MessageField('OSPolicyOSFilter', 2)
    resources = _messages.MessageField('OSPolicyResource', 3, repeated=True)