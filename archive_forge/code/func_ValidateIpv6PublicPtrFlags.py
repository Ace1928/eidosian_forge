from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def ValidateIpv6PublicPtrFlags(args):
    """Validates the values of IPv6 public PTR related flags."""
    network_interface = getattr(args, 'network_interface', None)
    if args.ipv6_public_ptr_domain is not None or args.no_ipv6_public_ptr:
        if network_interface is not None and network_interface != constants.DEFAULT_NETWORK_INTERFACE:
            raise compute_exceptions.ArgumentError("IPv6 Public PTR can only be enabled for default network interface '{0}' rather than '{1}'.".format(constants.DEFAULT_NETWORK_INTERFACE, network_interface))
    if args.ipv6_public_ptr_domain is not None and args.no_ipv6_public_ptr:
        raise exceptions.ConflictingArgumentsException('--ipv6-public-ptr-domain', '--no-ipv6-public-ptr')