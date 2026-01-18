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
def AddBillingProjectArg(parser, required=True, help_text=BILLING_HELP_TEST):
    """Add billing project argument to parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
    required: bool, whether to make this argument required.
    help_text: str, help text to overwrite the generic --billing-project help
      text.
  """
    parser.add_argument('--billing-project', required=required, help=help_text, action=actions.StoreProperty(properties.VALUES.billing.quota_project))