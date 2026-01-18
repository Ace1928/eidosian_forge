from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
import six
class PolicyNotFoundError(PolicyError):
    """Raised when the specified Ops Agents policy is not found."""

    def __init__(self, policy_id):
        message = 'Ops Agents policy [{policy_id}] not found.'.format(policy_id=policy_id)
        super(PolicyNotFoundError, self).__init__(message)