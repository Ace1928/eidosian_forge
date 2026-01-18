from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def ConstructCreateRequestFromArgsBeta(client, alloydb_messages, project_ref, args):
    """Validates command line input arguments and passes parent's resources for beta tracks.

  Args:
    client: Client for api_utils.py class.
    alloydb_messages: Messages module for the API client.
    project_ref: Parent resource path of the resource being created
    args: Command line input arguments.

  Returns:
    Fully-constructed request to create an AlloyDB instance.
  """
    instance_resource = _ConstructInstanceFromArgsBeta(client, alloydb_messages, args)
    return alloydb_messages.AlloydbProjectsLocationsClustersInstancesCreateRequest(instance=instance_resource, instanceId=args.instance, parent=project_ref.RelativeName())