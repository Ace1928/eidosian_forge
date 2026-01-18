from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class RequiredArgumentException(exceptions.Error):
    """An exception for when a usually optional argument is required in this case."""

    def __init__(self, parameter_name, message):
        super(RequiredArgumentException, self).__init__('Missing required argument [{0}]: {1}'.format(parameter_name, message))
        self.parameter_name = parameter_name