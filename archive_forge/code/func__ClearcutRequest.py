from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import platform
import socket
import time
from googlecloudsdk.command_lib.survey import question
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.survey import survey_check
from googlecloudsdk.core.util import platforms
from six.moves import http_client as httplib
def _ClearcutRequest(survey_instance):
    """Prepares clearcut LogRequest.

  Args:
     survey_instance: googlecloudsdk.command_lib.survey.survey.Survey, a survey
       object which contains user's response.

  Returns:
    A clearcut LogRequest object.
  """
    current_platform = platforms.Platform.Current()
    log_event = [{'source_extension_json': json.dumps(_ConcordEventForSurvey(survey_instance), sort_keys=True), 'event_time_ms': metrics.GetTimeMillis()}]
    return {'client_info': {'client_type': 'DESKTOP', 'desktop_client_info': {'os': current_platform.operating_system.id}}, 'log_source_name': 'CONCORD', 'zwieback_cookie': config.GetCID(), 'request_time_ms': metrics.GetTimeMillis(), 'log_event': log_event}