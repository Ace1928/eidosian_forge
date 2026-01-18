from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
def _create_or_modify_logging_config(job, args, messages):
    """Creates or modifies transfer LoggingConfig object based on args."""
    enable_posix_transfer_logs = getattr(args, 'enable_posix_transfer_logs', None)
    if not job.loggingConfig:
        if enable_posix_transfer_logs is None:
            job.loggingConfig = messages.LoggingConfig(enableOnpremGcsTransferLogs=True)
        else:
            job.loggingConfig = messages.LoggingConfig()
    if enable_posix_transfer_logs is not None:
        job.loggingConfig.enableOnpremGcsTransferLogs = enable_posix_transfer_logs
    if job.transferSpec.hdfsDataSource is not None:
        job.loggingConfig.enableOnpremGcsTransferLogs = False
    log_actions = getattr(args, 'log_actions', None)
    log_action_states = getattr(args, 'log_action_states', None)
    if not (log_actions or log_action_states):
        return
    existing_log_actions = job.loggingConfig and job.loggingConfig.logActions
    existing_log_action_states = job.loggingConfig and job.loggingConfig.logActionStates
    if not (log_actions and log_action_states) and (log_actions and (not existing_log_action_states) or (log_action_states and (not existing_log_actions))):
        raise ValueError('Both --log-actions and --log-action-states are required for a complete log config.')
    if log_actions:
        actions = []
        for action in log_actions:
            actions.append(getattr(job.loggingConfig.LogActionsValueListEntryValuesEnum, action.upper()))
        job.loggingConfig.logActions = actions
    if log_action_states:
        action_states = []
        for action_state in log_action_states:
            action_states.append(getattr(job.loggingConfig.LogActionStatesValueListEntryValuesEnum, action_state.upper()))
        job.loggingConfig.logActionStates = action_states