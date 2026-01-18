from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import flags
from googlecloudsdk.command_lib.dns import resource_args
from googlecloudsdk.command_lib.dns import util as command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _FetchResponsePolicyRule(response_policy, response_policy_rule, api_version, args):
    """Get response policy rule to be Updated."""
    client = util.GetApiClient(api_version)
    m = apis.GetMessagesModule('dns', api_version)
    get_request = m.DnsResponsePolicyRulesGetRequest(responsePolicy=response_policy, project=properties.VALUES.core.project.Get(), responsePolicyRule=response_policy_rule)
    if api_version == 'v2':
        get_request.location = args.location
    return client.responsePolicyRules.Get(get_request)