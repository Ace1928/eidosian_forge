from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def _ConstructInstanceFromArgsAlpha(client, alloydb_messages, args):
    """Validates command line input arguments and passes parent's resources to create an AlloyDB instance for alpha track.

  Args:
    client: Client for api_utils.py class.
    alloydb_messages: Messages module for the API client.
    args: Command line input arguments.

  Returns:
    An AlloyDB instance to create with the specified command line arguments.
  """
    instance_resource = _ConstructInstanceFromArgsBeta(client, alloydb_messages, args)
    return instance_resource