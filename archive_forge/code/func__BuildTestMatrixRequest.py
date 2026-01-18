from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import matrix_creator_common
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _BuildTestMatrixRequest(self, request_id):
    """Build a TestingProjectsTestMatricesCreateRequest for a test matrix.

    Args:
      request_id: {str} a unique ID for the CreateTestMatrixRequest.

    Returns:
      A TestingProjectsTestMatricesCreateRequest message.
    """
    spec = self._TestSpecFromType(self._args.type)
    return self._messages.TestingProjectsTestMatricesCreateRequest(projectId=self._project, testMatrix=self._BuildTestMatrix(spec), requestId=request_id)