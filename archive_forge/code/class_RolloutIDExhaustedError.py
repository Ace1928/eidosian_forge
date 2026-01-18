from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class RolloutIDExhaustedError(exceptions.Error):
    """Error when there are too many rollouts for a given release."""

    def __init__(self, release_name):
        super(RolloutIDExhaustedError, self).__init__('Rollout name space exhausted in release {}. Use --rollout-id to specify rollout ID.'.format(release_name))