from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddReplicationDestinationVolumeParametersArg(parser):
    """Adds the Destination Volume Parameters (--destination-volume-parameters) arg to the given parser.

  Args:
    parser: Argparse parser.
  """
    destination_volume_parameters_spec = {'storage_pool': str, 'volume_id': str, 'share_name': str, 'description': str}
    destination_volume_parameters_help = '      '
    parser.add_argument('--destination-volume-parameters', type=arg_parsers.ArgDict(spec=destination_volume_parameters_spec, required_keys=['storage_pool']), required=True, help=destination_volume_parameters_help)