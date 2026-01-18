from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ArgsForJobRefs(parser, **kwargs):
    """Register flags for specifying jobs using positional job IDs.

  Args:
    parser: The argparse.ArgParser to configure with job ID arguments.
    **kwargs: Extra arguments to pass to the add_argument call.
  """
    parser.add_argument('jobs', metavar='JOB_ID', help='Job IDs to operate on.', **kwargs)
    parser.add_argument('--region', metavar='REGION_ID', help="Region ID of the jobs' regional endpoint. " + dataflow_util.DEFAULT_REGION_MESSAGE)