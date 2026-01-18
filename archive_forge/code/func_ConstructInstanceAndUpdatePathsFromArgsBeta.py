from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def ConstructInstanceAndUpdatePathsFromArgsBeta(alloydb_messages, instance_ref, args):
    """Validates command line arguments and creates the instance and update paths for beta track.

  Args:
    alloydb_messages: Messages module for the API client.
    instance_ref: parent resource path of the resource being updated
    args: Command line input arguments.

  Returns:
    An AlloyDB instance and paths for update.
  """
    instance_resource, paths = ConstructInstanceAndUpdatePathsFromArgs(alloydb_messages, instance_ref, args)
    if args.update_mode:
        instance_resource.updatePolicy = alloydb_messages.UpdatePolicy(mode=_ParseUpdateMode(alloydb_messages, args.update_mode))
        update_mode_path = 'updatePolicy.mode'
        paths.append(update_mode_path)
    return (instance_resource, paths)