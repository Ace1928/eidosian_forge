from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class TestLabInfrastructureError(TestingError):
    """Encountered a Firebase Test Lab infrastructure error during testing."""

    def __init__(self, error):
        super(TestLabInfrastructureError, self).__init__('Firebase Test Lab infrastructure failure: {0}'.format(error), exit_code=exit_code.INFRASTRUCTURE_ERR)