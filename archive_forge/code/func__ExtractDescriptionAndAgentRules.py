from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.calliope import exceptions
def _ExtractDescriptionAndAgentRules(guest_policy_description):
    """Extract Ops Agents policy's description and agent rules.

  Extract Ops Agents policy's description and agent rules from description of
  OS Config guest policy.

  Args:
    guest_policy_description: OS Config guest policy's description.

  Returns:
    extracted description and agent rules for ops agents policy.

  Raises:
    BadArgumentException: If guest policy's description is illformed JSON
    object, or if it does not have keys description or agentRules.
  """
    try:
        decode_description = json.loads(guest_policy_description)
    except ValueError as e:
        raise exceptions.BadArgumentException('description', 'description field is not a JSON object: {}'.format(e))
    if not isinstance(decode_description, dict):
        raise exceptions.BadArgumentException('description', 'description field is not a JSON object.')
    try:
        decoded_description = decode_description['description']
    except KeyError as e:
        raise exceptions.BadArgumentException('description.description', 'missing a required key description: %s' % e)
    try:
        decoded_agent_rules = decode_description['agentRules']
    except KeyError as e:
        raise exceptions.BadArgumentException('description.agentRules', 'missing a required key agentRules: %s' % e)
    return (decoded_description, decoded_agent_rules)