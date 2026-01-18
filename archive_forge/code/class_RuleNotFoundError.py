from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.routers.nats.rules import flags
from googlecloudsdk.core import exceptions as core_exceptions
import six
class RuleNotFoundError(core_exceptions.Error):
    """Raised when a Rule is not found."""

    def __init__(self, rule_number):
        msg = 'Rule `{0}` not found'.format(rule_number)
        super(RuleNotFoundError, self).__init__(msg)