from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
class InvalidLogFormatException(DebugError):
    """A log format expression was invalid."""

    def __init__(self, message):
        super(InvalidLogFormatException, self).__init__(message)