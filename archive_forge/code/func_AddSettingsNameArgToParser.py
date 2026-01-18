from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.args import common_args
def AddSettingsNameArgToParser(parser):
    """Adds argument for the settings name to the parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('setting_name', metavar='SETTING_NAME', help='Name of the resource settings. The list of available settings can be fetched using the list command: \n $ gcloud resource-settings list')