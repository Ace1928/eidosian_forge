from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def DictToEnvVarsProperty(env_vars_type_class=None, env_vars=None):
    """Sets environment variables.

  Args:
    env_vars_type_class: type class of environment variables
    env_vars: a dict of environment variables

  Returns:
    An message with the environment variables from env_vars
  """
    if not env_vars_type_class or not env_vars:
        return None
    return env_vars_type_class(additionalProperties=[env_vars_type_class.AdditionalProperty(key=key, value=value) for key, value in sorted(env_vars.items())])