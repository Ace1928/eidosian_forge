from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InvalidArgumentException(exceptions.Error):
    """InvalidArgumentException is for malformed arguments."""

    def __init__(self, parameter_name, message):
        """Creates InvalidArgumentException.

    Args:
      parameter_name: str, the parameter flag or argument name
      message: str, the exception message
    """
        super(InvalidArgumentException, self).__init__('Invalid value for [{0}]: {1}'.format(parameter_name, message))
        self.parameter_name = parameter_name