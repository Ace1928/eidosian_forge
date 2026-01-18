from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def AddResourcePoliciesArgs(parser, action, resource, required=False):
    """Adds arguments related to resource policies."""
    if resource == 'instance-template':
        help_text = 'A list of resource policy names (not URLs) to be {action} each instance created using this instance template. If you attach any resource policies to an instance template, you can only use that instance template to create instances that are in the same region as the resource policies. Do not include resource policies that are located in different regions in the same instance template.'
    else:
        help_text = 'A list of resource policy names to be {action} the {resource}. The policies must exist in the same region as the {resource}.'
    parser.add_argument('--resource-policies', metavar='RESOURCE_POLICY', type=arg_parsers.ArgList(), required=required, help=help_text.format(action=action, resource=resource))