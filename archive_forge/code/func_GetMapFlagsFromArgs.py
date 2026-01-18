from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def GetMapFlagsFromArgs(flag_name, args):
    """Get the flags for updating this map and return their values in a dict.

  Args:
    flag_name: The base name of the flags
    args: The argparse namespace

  Returns:
    A dict of the flag values
  """
    specified_args = args.GetSpecifiedArgs()
    return {'set_flag_value': specified_args.get('--set-{}'.format(flag_name)), 'update_flag_value': specified_args.get('--update-{}'.format(flag_name)), 'clear_flag_value': specified_args.get('--clear-{}'.format(flag_name)), 'remove_flag_value': specified_args.get('--remove-{}'.format(flag_name)), 'file_flag_value': specified_args.get('--{}-file'.format(flag_name))}