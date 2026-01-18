from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import platform
import re
import time
import uuid
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
def _LogResponse(response, data):
    """Logs a response."""
    redact_resp_body_reason = data['redact_resp_body_reason']
    time_taken = time.time() - data['start_time']
    log.status.Print('---- response start ----')
    log.status.Print('status: {0}'.format(response.status_code))
    log.status.Print('-- headers start --')
    for h, v in sorted(six.iteritems(response.headers)):
        log.status.Print('{0}: {1}'.format(h, v))
    log.status.Print('-- headers end --')
    log.status.Print('-- body start --')
    if streaming_response_body:
        log.status.Print('<streaming body>')
    elif redact_resp_body_reason is None:
        log.status.Print(response.body)
    else:
        log.status.Print('Body redacted: {}'.format(redact_resp_body_reason))
    log.status.Print('-- body end --')
    log.status.Print('total round trip time (request+response): {0:.3f} secs'.format(time_taken))
    log.status.Print('---- response end ----')
    log.status.Print('----------------------')