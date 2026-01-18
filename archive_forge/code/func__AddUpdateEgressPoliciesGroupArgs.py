from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import acm_printer
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import levels
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def _AddUpdateEgressPoliciesGroupArgs(parser, api_version):
    """Add args for set/clear egress policies."""
    group_help = 'These flags modify the enforced EgressPolicies of this ServicePerimeter.'
    group = parser.add_mutually_exclusive_group(group_help)
    set_egress_policies_help_text = 'Path to a file containing a list of Egress Policies.\n\nThis file contains a list of YAML-compliant objects representing Egress Policies described in the API reference.\n\nFor more information about the alpha version, see:\nhttps://cloud.google.com/access-context-manager/docs/reference/rest/v1alpha/accessPolicies.servicePerimeters\nFor more information about non-alpha versions, see: \nhttps://cloud.google.com/access-context-manager/docs/reference/rest/v1/accessPolicies.servicePerimeters'
    set_egress_policies_arg = base.Argument('--set-egress-policies', metavar='YAML_FILE', help=set_egress_policies_help_text, type=ParseEgressPolicies(api_version))
    clear_egress_policies_help_text = 'Empties existing enforced Egress Policies.'
    clear_egress_policies_arg = base.Argument('--clear-egress-policies', help=clear_egress_policies_help_text, action='store_true')
    set_egress_policies_arg.AddToParser(group)
    clear_egress_policies_arg.AddToParser(group)