from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding as apitools_encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
import six
def RepositoryToCleanupPoliciesResponse(response, unused_args):
    """Formats Cleanup Policies output and displays Dry Run status."""
    if response.cleanupPolicyDryRun:
        log.status.Print('Dry run is enabled.')
    else:
        log.status.Print('Dry run is disabled.')
    if not response.cleanupPolicies:
        return []
    policies = apitools_encoding.MessageToDict(response.cleanupPolicies)
    sorted_policies = sorted(policies.values(), key=lambda p: p['id'])
    for policy in sorted_policies:
        policy['name'] = policy['id']
        del policy['id']
        policy['action'] = {'type': policy['action']}
    return sorted_policies