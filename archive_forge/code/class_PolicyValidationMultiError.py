from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
import six
class PolicyValidationMultiError(PolicyValidationError):
    """Raised when multiple Ops Agents policy validations fail."""

    def __init__(self, errors):
        super(PolicyValidationMultiError, self).__init__(' | '.join((six.text_type(error) for error in errors)))
        self.errors = errors