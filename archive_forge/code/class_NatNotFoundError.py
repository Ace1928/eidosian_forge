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
class NatNotFoundError(core_exceptions.Error):
    """Raised when a NAT is not found."""

    def __init__(self, name):
        self.name = name
        msg = 'NAT `{0}` not found'.format(name)
        super(NatNotFoundError, self).__init__(msg)