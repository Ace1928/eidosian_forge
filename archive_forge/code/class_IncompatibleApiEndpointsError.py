from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class IncompatibleApiEndpointsError(TestingError):
    """Two or more API endpoint overrides are incompatible with each other."""

    def __init__(self, endpoint1, endpoint2):
        super(IncompatibleApiEndpointsError, self).__init__('Service endpoints [{0}] and [{1}] are not compatible.'.format(endpoint1, endpoint2))