from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnet_flags
from googlecloudsdk.command_lib.compute.routers.nats import flags as nat_flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
class IpAllocationUnspecifiedError(core_exceptions.Error):
    """Raised when IP Allocation option is not specified for public NAT."""

    def __init__(self):
        msg = 'Either --nat-external-ip-pool or --auto-allocate-nat-external-ips must be specified for Public NAT.'
        super(IpAllocationUnspecifiedError, self).__init__(msg)