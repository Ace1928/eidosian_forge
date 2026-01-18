from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InvalidFlagValueError(exceptions.Error):
    """An error raised for an invalid value for certain flags."""

    def __init__(self, msg):
        message = 'Invalid flag value: {}'.format(msg)
        super(InvalidFlagValueError, self).__init__(message)