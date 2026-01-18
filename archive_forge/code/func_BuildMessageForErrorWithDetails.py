from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import batch
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.compute import operation_quota_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
def BuildMessageForErrorWithDetails(json_data):
    if operation_quota_utils.IsJsonOperationQuotaError(json_data.get('error', {})):
        return operation_quota_utils.CreateOperationQuotaExceededMsg(json_data)
    else:
        return json_data.get('error', {}).get('message')