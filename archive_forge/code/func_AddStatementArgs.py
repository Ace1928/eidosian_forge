from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddStatementArgs(parser):
    """Adds the sql statement."""
    parser.add_argument('--statement', help='A BigQuery Standard SQL compatible statement. If the query execution finishes within timeout and there is no pagination, the full query results will be returned. Otherwise, pass job_reference from previous call as `--job-referrence` to obtain the full results.')