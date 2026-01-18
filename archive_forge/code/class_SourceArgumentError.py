from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
class SourceArgumentError(calliope_exceptions.InvalidArgumentException):
    """Exception for errors related to using the --source argument."""

    def __init__(self, message):
        super(SourceArgumentError, self).__init__('--source', message)