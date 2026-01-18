from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.command_lib.util.args import map_util
import six
def AddUpdateEnvVarsFlags(parser):
    """Add flags for setting and removing environment variables.

  Args:
    parser: The argument parser.
  """
    map_util.AddUpdateMapFlags(parser, 'env-vars', long_name='environment variables', key_type=EnvVarKeyType, value_type=EnvVarValueType)