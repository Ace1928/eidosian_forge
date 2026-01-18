from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker as tracker
from googlecloudsdk.core.util import retry
def ToOperationResponse(message):
    raw_dict = encoding.MessageToDict(message)
    return encoding.DictToMessage(raw_dict, OperationsMessages().Operation.ResponseValue)