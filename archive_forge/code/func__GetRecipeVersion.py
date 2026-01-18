from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _GetRecipeVersion(prev_recipes, recipe_name):
    for recipe in prev_recipes or []:
        if recipe.name.startswith(recipe_name):
            return str(int(recipe.version) + 1)
    return '0'