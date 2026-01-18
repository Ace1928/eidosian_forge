from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreateRecipe(messages, agent_rule, os_type, prev_recipes):
    """Create a recipe for one agent rule in guest policy.

  Args:
    messages: os config guest policy api messages.
    agent_rule: ops agent policy agent rule.
    os_type: ops agent policy os type.
    prev_recipes: a list of original SoftwareRecipe.


  Returns:
    One software recipe in guest policy. If the package state is "removed", this
    software recipe has an empty run script. We still keep the software recipe
    to maintain versioning of the software recipe as the policy gets updated.
  """
    version = _GetRecipeVersion(prev_recipes, _AGENT_RULE_TEMPLATES[agent_rule.type].recipe_name)
    return messages.SoftwareRecipe(desiredState=messages.SoftwareRecipe.DesiredStateValueValuesEnum.UPDATED, installSteps=[_CreateStepInScript(messages, agent_rule, os_type)], name='%s-%s' % (_AGENT_RULE_TEMPLATES[agent_rule.type].recipe_name, version), version=version)