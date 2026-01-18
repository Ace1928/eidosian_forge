from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import re
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.api_lib.logging import util as logging_util
from googlecloudsdk.core import log as logging
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def _PayloadFormatter(log):
    """Used as a formatter for logs_util.LogPrinter.

  If the log has a JSON payload or a proto payload, the payloads will be
  JSON-ified. The text payload will be returned as-is.

  Args:
    log: the log entry to serialize to json

  Returns:
    A JSON serialization of a log's payload.
  """
    if hasattr(log, 'protoPayload') and log.protoPayload:
        return _PayloadToJSON(log.protoPayload)
    elif hasattr(log, 'textPayload') and log.textPayload:
        return log.textPayload
    elif hasattr(log, 'jsonPayload') and log.jsonPayload:
        return _PayloadToJSON(log.jsonPayload, is_json_payload=True)
    return 'No contents found.'