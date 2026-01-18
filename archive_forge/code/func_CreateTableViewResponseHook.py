from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.compute.os_config import flags
from googlecloudsdk.core import properties
import six
def CreateTableViewResponseHook(inventory_list, args):
    """Create ListTableRow from ListInventory response.

  Args:
    inventory_list: Response from ListInventory
    args: gcloud invocation args

  Returns:
    ListTableRow
  """
    view = args.view if args.view else 'basic'
    rows = []
    for inventory in inventory_list:
        installed_packages = 0
        available_packages = 0
        if view == 'full' and inventory.items:
            for v in six.itervalues(encoding.MessageToDict(inventory.items)):
                if 'installedPackage' in v:
                    installed_packages += 1
                elif 'availablePackage' in v:
                    available_packages += 1
        rows.append(ListTableRow(instance_id=inventory.name.split('/')[-2], instance_name=inventory.osInfo.hostname, os_long_name=inventory.osInfo.longName, installed_packages=installed_packages, available_packages=available_packages, update_time=inventory.updateTime, osconfig_agent_version=inventory.osInfo.osconfigAgentVersion))
    return {view: rows}