from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class TestExecutionNotFoundError(TestingError):
    """A test execution ID was not found within a test matrix."""

    def __init__(self, execution_id, matrix_id):
        super(TestExecutionNotFoundError, self).__init__('Test execution [{e}] not found in matrix [{m}].'.format(e=execution_id, m=matrix_id))