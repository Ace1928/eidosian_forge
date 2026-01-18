from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.functions import secrets_config
import six
def SecretEnvVarsToMessages(secret_env_vars_dict, messages):
    """Converts secrets from dict to cloud function SecretEnvVar message list.

  Args:
    secret_env_vars_dict: Secret environment variables configuration dict.
      Prefers a sorted ordered dict for consistency.
    messages: The GCF messages module to use.

  Returns:
    A list of cloud function SecretEnvVar message.
  """
    secret_environment_variables = []
    for secret_env_var_key, secret_env_var_value in six.iteritems(secret_env_vars_dict):
        secret_ref = _ParseSecretRef(secret_env_var_value)
        secret_environment_variables.append(messages.SecretEnvVar(key=secret_env_var_key, projectId=secret_ref['project'], secret=secret_ref['secret'], version=secret_ref['version']))
    return secret_environment_variables