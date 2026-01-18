from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.command_lib.util.args import map_util
import six
def BuildEnvVarKeyType(key):
    """Validator for build environment variable keys.

  All existing validations for environment variables are also applicable for
  build environment variables.

  Args:
    key: The build environment variable key.

  Returns:
    The build environment variable key type.
  Raises:
    ArgumentTypeError: If the key is not valid.
  """
    if key in ['GOOGLE_ENTRYPOINT', 'GOOGLE_FUNCTION_TARGET', 'GOOGLE_RUNTIME', 'GOOGLE_RUNTIME_VERSION']:
        raise argparse.ArgumentTypeError('{} is reserved for internal use by GCF deployments and cannot be used.'.format(key))
    return EnvVarKeyType(key)