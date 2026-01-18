from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import name_generator
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags
import ipaddr
from six.moves import zip  # pylint: disable=redefined-builtin
def CheckPurposeInSubnetwork(self, messages, purpose):
    if purpose != messages.Address.PurposeValueValuesEnum.GCE_ENDPOINT and purpose != messages.Address.PurposeValueValuesEnum.SHARED_LOADBALANCER_VIP:
        raise exceptions.InvalidArgumentException('--purpose', 'must be GCE_ENDPOINT or SHARED_LOADBALANCER_VIP for regional internal addresses.')