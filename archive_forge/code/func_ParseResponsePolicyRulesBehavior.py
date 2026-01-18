from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.dns import flags
import ipaddr
def ParseResponsePolicyRulesBehavior(args, version='v1'):
    """Parses response policy rule behavior."""
    m = GetMessages(version)
    if args.behavior == 'bypassResponsePolicy':
        return m.ResponsePolicyRule.BehaviorValueValuesEnum.BYPASS_RESPONSE_POLICY if version == 'v2' else m.ResponsePolicyRule.BehaviorValueValuesEnum.bypassResponsePolicy
    else:
        return None