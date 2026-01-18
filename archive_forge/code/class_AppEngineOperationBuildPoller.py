from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import enum
from googlecloudsdk.api_lib.app import exceptions as app_exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import requests
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
class AppEngineOperationBuildPoller(AppEngineOperationPoller):
    """Waits for a build to be present, or for the operation to finish."""

    def __init__(self, operation_service, operation_metadata_type):
        """Sets up poller for appengine operations.

    Args:
      operation_service: apitools.base.py.base_api.BaseApiService, api service
        for retrieving information about ongoing operation.
      operation_metadata_type: Message class for the Operation metadata (for
        instance, OperationMetadataV1, or OperationMetadataV1Beta).
    """
        super(AppEngineOperationBuildPoller, self).__init__(operation_service, operation_metadata_type)

    def IsDone(self, operation):
        if GetBuildFromOperation(operation, self.operation_metadata_type):
            return True
        return super(AppEngineOperationBuildPoller, self).IsDone(operation)