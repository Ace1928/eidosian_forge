from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.command_lib.util.args import map_util
import six
def EnvVarKeyType(key):
    """Validator for environment variable keys.

  Args:
    key: The environment variable key.

  Returns:
    The environment variable key.
  Raises:
    ArgumentTypeError: If the key is not a valid environment variable key.
  """
    if not key:
        raise argparse.ArgumentTypeError('Environment variable keys cannot be empty.')
    if key.startswith('X_GOOGLE_'):
        raise argparse.ArgumentTypeError('Environment variable keys that start with `X_GOOGLE_` are reserved for use by deployment tools and cannot be specified manually.')
    if '=' in key:
        raise argparse.ArgumentTypeError('Environment variable keys cannot contain `=`.')
    return key