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
def AddPublicPtrArgs(parser, instance=True):
    """Adds public PTR arguments for instance or access configuration."""
    public_ptr_args = parser.add_mutually_exclusive_group()
    if instance:
        no_public_ptr_help = '        If provided, no DNS PTR record is created for the external IP of the\n        instance. Mutually exclusive with public-ptr-domain.\n        '
    else:
        no_public_ptr_help = '        If provided, no DNS PTR record is created for the external IP in the\n        access configuration. Mutually exclusive with public-ptr-domain.\n        '
    public_ptr_args.add_argument('--no-public-ptr', action='store_true', help=no_public_ptr_help)
    if instance:
        public_ptr_help = '        Creates a DNS PTR record for the external IP of the instance.\n        '
    else:
        public_ptr_help = '        Creates a DNS PTR record for the external IP in the access\n        configuration. This option can only be specified for the default\n        network-interface, "nic0".'
    public_ptr_args.add_argument('--public-ptr', action='store_true', help=public_ptr_help)
    public_ptr_domain_args = parser.add_mutually_exclusive_group()
    if instance:
        no_public_ptr_domain_help = '        If both this flag and --public-ptr are specified, creates a DNS PTR\n        record for the external IP of the instance with the PTR domain name\n        being the DNS name of the instance.\n        '
    else:
        no_public_ptr_domain_help = '        If both this flag and --public-ptr are specified, creates a DNS PTR\n        record for the external IP in the access configuration with the PTR\n        domain name being the DNS name of the instance.\n        '
    public_ptr_domain_args.add_argument('--no-public-ptr-domain', action='store_true', help=no_public_ptr_domain_help)
    if instance:
        public_ptr_domain_help = '        Assigns a custom PTR domain for the external IP of the instance.\n        Mutually exclusive with no-public-ptr.\n        '
    else:
        public_ptr_domain_help = '        Assigns a custom PTR domain for the external IP in the access\n        configuration. Mutually exclusive with no-public-ptr. This option can\n        only be specified for the default network-interface, "nic0".\n        '
    public_ptr_domain_args.add_argument('--public-ptr-domain', help=public_ptr_domain_help)