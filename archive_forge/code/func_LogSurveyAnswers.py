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
def LogSurveyAnswers(survey_instance):
    """Sends survey response to clearcut table."""
    http_client = requests.GetSession()
    headers = {'user-agent': metrics.GetUserAgent()}
    body = json.dumps(_ClearcutRequest(survey_instance), sort_keys=True)
    response = http_client.request('POST', _CLEARCUT_ENDPOINT, data=body, headers=headers)
    if response.status_code != httplib.OK:
        raise SurveyNotRecordedError('We cannot record your feedback at this time, please try again later.')
    _UpdateSurveyCache()
    log.err.Print('Your response is submitted.')