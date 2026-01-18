from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_security.firewall_endpoints import activation_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddUpdateBillingProjectArg(parser, required=False, help_text=BILLING_HELP_TEST):
    """Add update billing project argument to parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
    required: bool, whether to make this argument required.
    help_text: str, help text to display on the --update-billing-project help
      text.
  """
    parser.add_argument('--update-billing-project', required=required, help=help_text, metavar='BILLING_PROJECT', action=actions.StoreProperty(properties.VALUES.billing.quota_project))