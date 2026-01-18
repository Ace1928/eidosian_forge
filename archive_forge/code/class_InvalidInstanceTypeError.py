from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InvalidInstanceTypeError(exceptions.Error):
    """Instance has the wrong environment."""

    def __init__(self, environment, message=None):
        msg = '{} instances do not support this operation.'.format(environment)
        if message:
            msg += '  ' + message
        super(InvalidInstanceTypeError, self).__init__(msg)