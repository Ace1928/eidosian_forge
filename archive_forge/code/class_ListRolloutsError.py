from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class ListRolloutsError(exceptions.Error):
    """Error when it failed to list the rollouts that belongs to a release."""

    def __init__(self, release_name):
        super(ListRolloutsError, self).__init__('Failed to list rollouts for {}.'.format(release_name))