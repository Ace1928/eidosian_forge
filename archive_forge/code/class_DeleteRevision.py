from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_ex
from googlecloudsdk.api_lib.pubsub import schemas
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import log
class DeleteRevision(base.Command):
    """Delete a Pub/Sub schema revision."""
    detailed_help = {'EXAMPLES': '          To roll back to an existing schema revision called "key-schema" with revision_id: "0a0b0c0d", run:\n                    \n$ {command} key-schema@0a0b0c0d\n          '}

    @staticmethod
    def Args(parser):
        resource_args.AddSchemaResourceArg(parser, 'revision to delete.')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      A serialized object (dict) describing the results of the operation.
      This description fits the Resource described in the ResourceRegistry under
      'pubsub.projects.schemas'.

    Raises:
      util.RequestFailedError: if any of the requests to the API failed.
    """
        client = schemas.SchemasClient()
        schema_ref = util.ParseSchemaName(args.schema)
        try:
            result = client.DeleteRevision(schema_ref=schema_ref)
        except api_ex.HttpError as error:
            exc = exceptions.HttpException(error)
            log.DeletedResource(schema_ref, kind='schema revision', failed=util.CreateFailureErrorMessage(exc.payload.status_message))
            return
        log.DeletedResource(result.name, kind='schema revision')
        return result