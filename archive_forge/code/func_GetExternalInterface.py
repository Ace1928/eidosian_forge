from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import collections
import datetime
import json
import os
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
from googlecloudsdk.core.util.files import FileReader
from googlecloudsdk.core.util.files import FileWriter
import six
def GetExternalInterface(instance_resource, no_raise=False):
    """Returns the network interface of the instance with an external IP address.

  Args:
    instance_resource: An instance resource object.
    no_raise: A boolean flag indicating whether or not to return None instead of
      raising.

  Raises:
    UnallocatedIPAddressError: If the instance_resource's external IP address
      has yet to be allocated.
    MissingExternalIPAddressError: If no external IP address is found for the
      instance_resource and no_raise is False.

  Returns:
    A network interface resource object or None if no_raise and a network
    interface with an external IP address does not exist.
  """
    no_ip = False
    if instance_resource.networkInterfaces:
        for network_interface in instance_resource.networkInterfaces:
            access_configs = network_interface.accessConfigs
            ipv6_access_configs = network_interface.ipv6AccessConfigs
            if access_configs:
                if access_configs[0].natIP:
                    return network_interface
                elif not no_raise:
                    no_ip = True
            if ipv6_access_configs:
                if ipv6_access_configs[0].externalIpv6:
                    return network_interface
                elif not no_raise:
                    no_ip = True
            if no_ip:
                raise UnallocatedIPAddressError('Instance [{0}] in zone [{1}] has not been allocated an external IP address yet. Try rerunning this command later.'.format(instance_resource.name, path_simplifier.Name(instance_resource.zone)))
    if no_raise:
        return None
    raise MissingExternalIPAddressError('Instance [{0}] in zone [{1}] does not have an external IP address, so you cannot SSH into it. To add an external IP address to the instance, use [gcloud compute instances add-access-config].'.format(instance_resource.name, path_simplifier.Name(instance_resource.zone)))