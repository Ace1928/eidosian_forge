from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags as security_policy_flags
from googlecloudsdk.command_lib.compute.security_policies.rules import flags
from googlecloudsdk.core import properties
@classmethod
def _ConvertRequestFieldToAdd(cls, compute_client, request_field_to_remove):
    """Converts RequestFieldToAdd."""
    request_field = compute_client.messages.SecurityPolicyRulePreconfiguredWafConfigExclusionFieldParams()
    op = request_field_to_remove.get('op') or ''
    if op:
        request_field.op = compute_client.messages.SecurityPolicyRulePreconfiguredWafConfigExclusionFieldParams.OpValueValuesEnum(op)
    val = request_field_to_remove.get('val') or ''
    if val:
        request_field.val = val
    return request_field