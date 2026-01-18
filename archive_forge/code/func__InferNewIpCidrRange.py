from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute.networks.subnets import flags
from googlecloudsdk.core.console import console_io
import ipaddress
import six
def _InferNewIpCidrRange(self, subnet_name, original_ip_cidr_range, new_prefix_length):
    unmasked_new_ip_range = '{0}/{1}'.format(original_ip_cidr_range.split('/')[0], new_prefix_length)
    network = ipaddress.IPv4Network(six.text_type(unmasked_new_ip_range), strict=False)
    return six.text_type(network)