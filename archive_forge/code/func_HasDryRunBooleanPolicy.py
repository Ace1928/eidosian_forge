from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.org_policies import arguments
from googlecloudsdk.command_lib.org_policies import utils
def HasDryRunBooleanPolicy(dry_run_spec):
    if dry_run_spec:
        return any([rule.enforce is not None for rule in dry_run_spec.rules])
    return False