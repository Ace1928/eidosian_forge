from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddSnapshotScheduleArgListToParser(parser, required=True):
    """Sets up an argument for a snapshot schedule."""
    spec = {'crontab_spec': str, 'retention_count': int, 'prefix': str}
    parser.add_argument('--schedule', required=required, type=arg_parsers.ArgDict(spec=spec, max_length=len(spec), required_keys=spec.keys()), action='append', metavar='CRONTAB_SPEC,RETENTION_COUNT,PREFIX', help='\n              Adds a schedule for taking snapshots of volumes under this policy.\n              This flag may be repeated to specify up to 5 schedules.\n\n              *crontab_spec*::: Specification of the times at which snapshots\n              will be taken. This should be in Crontab format:\n              http://en.wikipedia.org/wiki/Cron#Overview\n\n              *retention_count*::: The maximum number of snapshots to retain in\n              this schedule.\n\n              *prefix*::: Value to append to the name of snapshots created by\n              this schedule.\n\n           ')