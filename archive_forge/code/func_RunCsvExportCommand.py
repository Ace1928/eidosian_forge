from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import export_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def RunCsvExportCommand(args, client):
    """Exports data from a Cloud SQL instance to a CSV file.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.
    client: SqlClient instance, with sql_client and sql_messages props, for use
      in generating messages and making API calls.

  Returns:
    A dict object representing the operations resource describing the export
    operation if the export was successful.
  """
    csv_export_context = export_util.CsvExportContext(client.sql_messages, args.uri, args.database, args.query, offload=args.offload, quote=args.quote, escape=args.escape, fields_terminated_by=args.fields_terminated_by, lines_terminated_by=args.lines_terminated_by)
    if args.offload:
        log.status.write('Serverless exports cost extra. See the pricing page for more information: https://cloud.google.com/sql/pricing.\n')
    return RunExportCommand(args, client, csv_export_context)