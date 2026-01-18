from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudiot import devices
from googlecloudsdk.api_lib.cloudiot import registries
from googlecloudsdk.command_lib.iot import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import times
import six
def ReadConfigData(args):
    """Read configuration data from the parsed arguments.

  See command_lib.iot.flags for the flag definitions.

  Args:
    args: a parsed argparse Namespace object containing config_data and
      config_file.

  Returns:
    str, the binary configuration data

  Raises:
    ValueError: unless exactly one of --config-data, --config-file given
  """
    if args.IsSpecified('config_data') and args.IsSpecified('config_file'):
        raise ValueError('Both --config-data and --config-file given.')
    if args.IsSpecified('config_data'):
        return http_encoding.Encode(args.config_data)
    elif args.IsSpecified('config_file'):
        return files.ReadBinaryFileContents(args.config_file)
    else:
        raise ValueError('Neither --config-data nor --config-file given.')