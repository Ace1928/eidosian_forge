from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class RolloutCannotAdvanceError(exceptions.Error):
    """Error when a rollout cannot be advanced because of a failed precondition."""

    def __init__(self, rollout_name, failed_activity_msg):
        error_msg = '{} Rollout {} cannot be advanced.'.format(failed_activity_msg, rollout_name)
        super(RolloutCannotAdvanceError, self).__init__(error_msg)