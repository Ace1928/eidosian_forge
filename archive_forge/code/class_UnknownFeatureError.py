from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class UnknownFeatureError(exceptions.Error):
    """An error raised when information is requested for an unknown Feature."""

    def __init__(self, name):
        message = '{} is not a supported feature'.format(name)
        super(UnknownFeatureError, self).__init__(message)