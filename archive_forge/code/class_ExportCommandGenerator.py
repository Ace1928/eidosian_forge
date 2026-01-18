from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import sys
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpBadRequestError
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
import six
class ExportCommandGenerator(BaseCommandGenerator):
    """Generator for export commands."""
    command_type = yaml_command_schema.CommandType.EXPORT

    def _Generate(self):
        """Generates an export command.

    An export command has a single resource argument and an API method to call
    to get the resource. The result is exported to a local yaml file provided
    by the `--destination` flag, or to stdout if nothing is provided.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """
        from googlecloudsdk.command_lib.export import util as export_util

        class Command(base.ExportCommand):
            """Export command enclosure."""

            @staticmethod
            def Args(parser):
                self._CommonArgs(parser)
                parser.add_argument('--destination', help='\n            Path to a YAML file where the configuration will be exported.\n            The exported data will not contain any output-only fields.\n            Alternatively, you may omit this flag to write to standard output.\n            For a schema describing the export/import format, see\n            $CLOUDSDKROOT/lib/googlecloudsdk/schemas/...\n          ')

            def Run(self_, args):
                unused_ref, response = self._CommonRun(args)
                method = self.arg_generator.GetPrimaryResource(self.methods, args).method
                schema_path = export_util.GetSchemaPath(method.collection.api_name, self.spec.request.api_version, type(response).__name__)
                if args.IsSpecified('destination'):
                    with files.FileWriter(args.destination) as stream:
                        export_util.Export(message=response, stream=stream, schema_path=schema_path)
                    return log.status.Print("Exported [{}] to '{}'.".format(response.name, args.destination))
                else:
                    export_util.Export(message=response, stream=sys.stdout, schema_path=schema_path)
        return Command