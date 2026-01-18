from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
class ArgumentValidationError(exceptions.Error):
    """Raised when a user specifies --rules and --allow parameters together."""

    def __init__(self, error_message):
        super(ArgumentValidationError, self).__init__(error_message)