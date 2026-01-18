from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.org_policies import exceptions
def _GetConstraintNameTokens(constraint_name):
    """Returns the individual tokens from the constraint name.

  Args:
    constraint_name: The name of the constraint. A constraint name has the
      following syntax:
        [organizations|folders|projects]/{resource_id}/constraints/{constraint_name}.
  """
    constraint_name_tokens = constraint_name.split('/')
    if len(constraint_name_tokens) != 4:
        raise exceptions.InvalidInputError("Invalid constraint name '{}': Name must be in the form [projects|folders|organizations]/{{resource_id}}/constraints/{{constraint_name}}.".format(constraint_name))
    return constraint_name_tokens