from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.org_policies import arguments
from googlecloudsdk.command_lib.org_policies import utils
def PolicyOutput(is_policy_set, is_dry_run_policy_set):
    if is_policy_set and is_dry_run_policy_set:
        return 'LIVE_AND_DRY_RUN_SET'
    elif is_policy_set:
        return 'SET'
    elif is_dry_run_policy_set:
        return 'DRY_RUN_SET'
    return '-'