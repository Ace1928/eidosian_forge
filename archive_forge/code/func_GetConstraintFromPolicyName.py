from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.org_policies import exceptions
def GetConstraintFromPolicyName(policy_name):
    """Returns the constraint from the specified policy name.

  A constraint has the following syntax: constraints/{constraint_name}.

  Args:
    policy_name: The name of the policy. A policy name has the following syntax:
      [organizations|folders|projects]/{resource_id}/policies/{constraint_name}.
  """
    policy_name_tokens = _GetPolicyNameTokens(policy_name)
    return 'constraints/{}'.format(policy_name_tokens[3])