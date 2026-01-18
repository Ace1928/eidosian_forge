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
def GetExternalIPAddress(instance_resource, no_raise=False):
    """Returns the external IP address of the instance.

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
    A string IPv4 address or IPv6 address if the IPv4 address does not exit
    or None if no_raise is True and no external IP exists.
  """
    network_interface = GetExternalInterface(instance_resource, no_raise=no_raise)
    if network_interface:
        if hasattr(network_interface, 'accessConfigs') and network_interface.accessConfigs:
            return network_interface.accessConfigs[0].natIP
        elif hasattr(network_interface, 'ipv6AccessConfigs') and network_interface.ipv6AccessConfigs:
            return network_interface.ipv6AccessConfigs[0].externalIpv6