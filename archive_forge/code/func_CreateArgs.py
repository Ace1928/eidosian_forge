from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def CreateArgs(run_task_request, args):
    """Creates Args input compatible for creating a RunTaskRequest object."""
    if getattr(args, 'ARGS', None):
        args_ref = dataplex_api.FetchExecutionSpecArgs(args.ARGS)
        if len(args_ref) > 0:
            return run_task_request.ArgsValue(additionalProperties=[run_task_request.ArgsValue.AdditionalProperty(key=key, value=value) for key, value in sorted(args_ref.items())])
    return None