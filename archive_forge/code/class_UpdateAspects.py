from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import entry as entry_api
from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.dataplex import flags
from googlecloudsdk.command_lib.dataplex import resource_args
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class UpdateAspects(base.UpdateCommand):
    """Add or update aspects for a Dataplex Entry."""
    detailed_help = {'EXAMPLES': '\n          To add or update aspects for the Dataplex entry `entry1` within the entry group `entry-group1` in location `us-central1` from the YAML/JSON file, run:\n\n            $ {command} entry1 --project=test-project --location=us-central1 --entry-group entry-group1 --aspects=path-to-a-file-with-aspects.json\n\n          '}

    @staticmethod
    def Args(parser: parser_arguments.ArgumentInterceptor):
        resource_args.AddEntryResourceArg(parser)
        flags.AddAspectFlags(parser, update_aspects_name='aspects', remove_aspects_name=None, required=True)

    @gcloud_exception.CatchHTTPErrorRaiseHTTPException('Status code: {status_code}. {status_message}.')
    def Run(self, args: parser_extensions.Namespace):
        return entry_api.Update(args, update_aspects_arg_name='aspects')