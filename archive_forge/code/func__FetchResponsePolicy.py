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
def _FetchResponsePolicy(self, response_policy_ref, api_version):
    """Get response policy to be Updated."""
    client = util.GetApiClient(api_version)
    message_module = apis.GetMessagesModule('dns', api_version)
    get_request = message_module.DnsResponsePoliciesGetRequest(responsePolicy=response_policy_ref.Name(), project=response_policy_ref.project)
    return client.responsePolicies.Get(get_request)