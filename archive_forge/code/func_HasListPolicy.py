from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.org_policies import arguments
from googlecloudsdk.command_lib.org_policies import utils
def HasListPolicy(spec):
    if not spec:
        return False
    for rule in spec.rules:
        if rule.values is not None or rule.allowAll is not None or rule.denyAll is not None:
            return True
    return False