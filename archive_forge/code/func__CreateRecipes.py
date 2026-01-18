from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreateRecipes(messages, agent_rules, os_type, prev_recipes):
    """Create recipes in guest policy.

  Args:
    messages: os config guest policy api messages.
    agent_rules: ops agent policy agent rules.
    os_type: ops agent policy os_type.
    prev_recipes: a list of original SoftwareRecipe.

  Returns:
    Recipes in guest policy
  """
    recipes = []
    for agent_rule in agent_rules or []:
        recipes.append(_CreateRecipe(messages, agent_rule, os_type, prev_recipes))
    return recipes