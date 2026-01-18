from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.command_lib.run import commands
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _ByStartAndCreationTime(ex):
    """Sort key that sorts executions by start time, newest and unstarted first.

  All unstarted executions will be first and sorted by their creation timestamp,
  all started executions will be second and sorted by their start time.

  Args:
    ex: googlecloudsdk.api_lib.run.execution.Execution

  Returns:
    The lastTransitionTime of the Started condition or the creation timestamp of
    the execution if the execution is unstarted.
  """
    return (False if ex.started_condition and ex.started_condition['status'] is not None else True, ex.started_condition['lastTransitionTime'] if ex.started_condition and ex.started_condition['lastTransitionTime'] else ex.creation_timestamp)