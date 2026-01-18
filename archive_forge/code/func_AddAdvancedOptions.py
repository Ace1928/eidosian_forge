from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddAdvancedOptions(parser, required=False):
    """Adds the cloud armor advanced options arguments to the argparse."""
    parser.add_argument('--json-parsing', choices=['DISABLED', 'STANDARD', 'STANDARD_WITH_GRAPHQL'], type=lambda x: x.upper(), required=required, help='The JSON parsing behavior for this rule. Must be one of the following values: [DISABLED, STANDARD, STANDARD_WITH_GRAPHQL].')
    parser.add_argument('--json-custom-content-types', type=arg_parsers.ArgList(), metavar='CONTENT_TYPE', help="      A comma-separated list of custom Content-Type header values to apply JSON\n      parsing for preconfigured WAF rules. Only applicable when JSON parsing is\n      enabled, like ``--json-parsing=STANDARD''. When configuring a Content-Type\n      header value, only the type/subtype needs to be specified, and the\n      parameters should be excluded.\n      ")
    parser.add_argument('--log-level', choices=['NORMAL', 'VERBOSE'], type=lambda x: x.upper(), required=required, help='The level of detail to display for WAF logging.')
    parser.add_argument('--user-ip-request-headers', type=arg_parsers.ArgList(), metavar='USER_IP_REQUEST_HEADER', help="      A comma-separated list of request header names to use for resolving the\n      caller's user IP address.\n      ")