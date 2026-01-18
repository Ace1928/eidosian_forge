from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
def GetInstanceNetworkTitleString(instance):
    """Returns a string that identifies the instance.

  Args:
    instance: The instance proto.

  Returns:
    A string that identifies the zone and the external ip of the instance.
  """
    external_ip = ssh_utils.GetExternalIPAddress(instance)
    result = '[{instance_name}] ({instance_ip})'.format(instance_name=instance.selfLink, instance_ip=external_ip)
    return result