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
def DeleteCleanupPolicyFields(unused_ref, args, request):
    removed_policies = args.policynames.split(',')
    remaining_policies = []
    if request.repository.cleanupPolicies:
        for policy in request.repository.cleanupPolicies.additionalProperties:
            if policy.key not in removed_policies:
                remaining_policies.append(policy)
        request.repository.cleanupPolicies.additionalProperties = remaining_policies
    request.updateMask = None
    return request