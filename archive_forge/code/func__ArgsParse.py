from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding_helper
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.orgpolicy import utils as orgpolicy_utils
from googlecloudsdk.api_lib.policy_intelligence import orgpolicy_simulator
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.policy_intelligence.simulator.orgpolicy import utils
from googlecloudsdk.core import log
def _ArgsParse(parser):
    """Parses arguments for the commands."""
    parser.add_argument('--organization', metavar='ORGANIZATION_ID', required=True, help='Organization ID.')
    parser.add_argument('--policies', type=arg_parsers.ArgList(), metavar='POLICIES', action=arg_parsers.UpdateAction, help='Path to the JSON or YAML file that contains the Org Policy to simulate.\n      Multiple Policies can be simulated by providing multiple, comma-separated paths.\n      E.g. --policies=p1.json,p2.json.\n      The format of policy can be found and created by `gcloud org-policies set-policy`.\n      See https://cloud.google.com/sdk/gcloud/reference/org-policies/set-policy for more details.\n      ')
    parser.add_argument('--custom-constraints', type=arg_parsers.ArgList(), metavar='CUSTOM_CONSTRAINTS', action=arg_parsers.UpdateAction, help='Path to the JSON or YAML file that contains the Custom Constraints to simulate.\n      Multiple Custom Constraints can be simulated by providing multiple, comma-separated paths.\n      e.g., --custom-constraints=constraint1.json,constraint2.json.\n      ')