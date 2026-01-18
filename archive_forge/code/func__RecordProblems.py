from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import batch_helper
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import single_request_helper
from googlecloudsdk.api_lib.util import exceptions as http_exceptions
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _RecordProblems(operation, warnings, errors):
    """Records any warnings and errors into the given lists."""
    for warning in operation.warnings or []:
        warnings.append(warning.message)
    if operation.error:
        for error in operation.error.errors or []:
            errors.append((operation.httpErrorStatusCode, error))