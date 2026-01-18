from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
def _is_agent_pool_still_creating(result, retryer_state):
    """Takes AgentPool Apitools object and returns if it's state is "creating".

  When an AgentPool create request is sent to the backend, it takes a few
  moments for the pool's state to go from CREATING to CREATED. This check
  is useful to see if we can start acting like the pool exists yet.

  Args:
    result (messages.AgentPool): Object representing current state of AgentPool
      on the backend.
    retryer_state (retry.RetryerState): Unused. Contains info about trials and
      time passed.

  Returns:
    Boolean representing if AgentPool's state is "CREATING." False = "CREATED".
  """
    del retryer_state
    messages = apis.GetMessagesModule('transfer', 'v1')
    return result.state == messages.AgentPool.StateValueValuesEnum.CREATING