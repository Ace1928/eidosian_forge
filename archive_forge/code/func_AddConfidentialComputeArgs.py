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
def AddConfidentialComputeArgs(parser, support_confidential_compute_type=False, support_confidential_compute_type_tdx=False) -> None:
    """Adds flags for confidential compute for instance."""
    if support_confidential_compute_type:
        choices = {'SEV': 'Secure Encrypted Virtualization', 'SEV_SNP': 'Secure Encrypted Virtualization - Secure Nested Paging'}
        help_text = '        The instance boots with Confidential Computing enabled. Confidential\n        Computing can be based on Secure Encrypted Virtualization (SEV) or Secure\n        Encrypted Virtualization - Secure Nested Paging (SEV-SNP), both of which\n        are AMD virtualization features for running confidential instances.\n        '
        if support_confidential_compute_type_tdx:
            choices['TDX'] = 'Trust Domain eXtension'
            help_text = ''.join((help_text, '        Trust Domain eXtension based on Intel virtualization features for\n        running confidential instances is also supported.\n        '))
        parser = parser.add_mutually_exclusive_group()
        parser.add_argument('--confidential-compute-type', dest='confidential_compute_type', choices=choices, help=help_text)
        bool_flag_action = actions.DeprecationAction('--confidential-compute', warn='The --confidential-compute flag will soon be deprecated. Please use `--confidential-compute-type=SEV` instead', action='store_true')
    else:
        bool_flag_action = 'store_true'
    parser.add_argument('--confidential-compute', dest='confidential_compute', action=bool_flag_action, default=None, help='      The instance boots with Confidential Computing enabled. Confidential\n      Computing is based on Secure Encrypted Virtualization (SEV), an AMD\n      virtualization feature for running confidential instances.\n      ')