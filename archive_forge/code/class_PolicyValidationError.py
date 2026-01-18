from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
import six
class PolicyValidationError(PolicyError):
    """Raised when Ops Agents policy validation fails."""