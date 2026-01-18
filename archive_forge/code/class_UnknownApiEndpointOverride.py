from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class UnknownApiEndpointOverride(exceptions.Error):
    """Class for errors by unknown endpoint override."""

    def __init__(self, api_name):
        message = 'Unknown api_endpoint_overrides value for {}'.format(api_name)
        super(UnknownApiEndpointOverride, self).__init__(message)