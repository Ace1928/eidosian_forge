from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class RedeployRolloutError(exceptions.Error):
    """Error when a rollout can't be redeployed.

  Redeploy can only be used for rollouts that are in a SUCCEEDED or FAILED
  state.
  """

    def __init__(self, target_name, rollout_name, rollout_state):
        error_msg = "Unable to redeploy target {}. Rollout {} is in state {} that can't be redeployed".format(target_name, rollout_name, rollout_state)
        super(RedeployRolloutError, self).__init__(error_msg)