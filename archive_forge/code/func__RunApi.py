from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import time
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def _RunApi(self, args, env_ref):
    cmd_params = []
    if args.tree:
        subcommand = 'pipdeptree'
        cmd_params.append('--warn')
    else:
        subcommand = 'pip list'
    execute_result = environments_api_util.ExecuteAirflowCommand(command='list-packages', subcommand=subcommand, parameters=[], environment_ref=env_ref, release_track=self.ReleaseTrack())
    if not execute_result.executionId:
        raise command_util.Error('Cannot execute subcommand for environment. Got empty execution Id.')
    output_end = False
    next_line = 1
    wait_time_seconds = DEFAULT_POLL_TIME_SECONDS
    poll_result = None
    cur_consequetive_poll_errors = 0
    while not output_end:
        lines = None
        try:
            time.sleep(wait_time_seconds + random.uniform(-POLL_JITTER_SECONDS, POLL_JITTER_SECONDS))
            poll_result = environments_api_util.PollAirflowCommand(execution_id=execute_result.executionId, pod_name=execute_result.pod, pod_namespace=execute_result.podNamespace, next_line_number=next_line, environment_ref=env_ref, release_track=self.ReleaseTrack())
            cur_consequetive_poll_errors = 0
            output_end = poll_result.outputEnd
            lines = poll_result.output
            lines.sort(key=lambda line: line.lineNumber)
        except:
            cur_consequetive_poll_errors += 1
        if cur_consequetive_poll_errors == MAX_CONSECUTIVE_POLL_ERRORS:
            raise command_util.Error('Cannot fetch list-packages command status.')
        if not lines:
            wait_time_seconds = min(wait_time_seconds * EXP_BACKOFF_MULTIPLIER, MAX_POLL_TIME_SECONDS)
        else:
            wait_time_seconds = DEFAULT_POLL_TIME_SECONDS
            for line in lines:
                log.Print(line.content if line.content else '')
            next_line = lines[-1].lineNumber + 1
    if poll_result and poll_result.exitInfo and poll_result.exitInfo.exitCode:
        log.error('Command exit code: {}'.format(poll_result.exitInfo.error))
        exit(poll_result.exitInfo.exitCode)