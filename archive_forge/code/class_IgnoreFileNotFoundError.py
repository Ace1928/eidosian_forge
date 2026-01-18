from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
class IgnoreFileNotFoundError(calliope_exceptions.InvalidArgumentException):
    """Exception for when file specified by --ignore-file is not found."""

    def __init__(self, message):
        super(IgnoreFileNotFoundError, self).__init__('--ignore-file', message)