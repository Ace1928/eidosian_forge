from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import instances as command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def RunBaseCloneCommand(args, release_track):
    """Clones a Cloud SQL instance.

  Args:
    args: argparse.Namespace, The arguments used to invoke this command.
    release_track: base.ReleaseTrack, the release track that this was run under.

  Returns:
    A dict object representing the operations resource describing the
    clone operation if the clone was successful.
  Raises:
    ArgumentError: The arguments are invalid for some reason.
  """
    client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    source_instance_ref, destination_instance_ref = _GetInstanceRefsFromArgs(args, client)
    request = sql_messages.SqlInstancesCloneRequest(project=source_instance_ref.project, instance=source_instance_ref.instance, instancesCloneRequest=sql_messages.InstancesCloneRequest(cloneContext=sql_messages.CloneContext(kind='sql#cloneContext', destinationInstanceName=destination_instance_ref.instance)))
    _UpdateRequestFromArgs(request, args, sql_messages, release_track)
    try:
        source_instance_resource = sql_client.instances.Get(sql_messages.SqlInstancesGetRequest(project=source_instance_ref.project, instance=source_instance_ref.instance))
        if source_instance_resource.diskEncryptionConfiguration:
            command_util.ShowCmekWarning('clone', 'the source instance')
    except apitools_exceptions.HttpError:
        pass
    result = sql_client.instances.Clone(request)
    operation_ref = client.resource_parser.Create('sql.operations', operation=result.name, project=destination_instance_ref.project)
    if args.async_:
        if not args.IsSpecified('format'):
            args.format = 'default'
        return sql_client.operations.Get(sql_messages.SqlOperationsGetRequest(project=operation_ref.project, operation=operation_ref.operation))
    operations.OperationsV1Beta4.WaitForOperation(sql_client, operation_ref, 'Cloning Cloud SQL instance')
    log.CreatedResource(destination_instance_ref)
    rsource = sql_client.instances.Get(sql_messages.SqlInstancesGetRequest(project=destination_instance_ref.project, instance=destination_instance_ref.instance))
    rsource.kind = None
    return rsource