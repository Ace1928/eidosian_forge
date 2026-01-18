from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr_os
import six
from six.moves import range
from six.moves import urllib
def OpenNewIssueInBrowser(info, log_data):
    """Opens a new tab in the web browser to the new issue page for Cloud SDK.

  The page will be pre-populated with relevant information.

  Args:
    info: InfoHolder, the data from of `gcloud info`
    log_data: LogData, parsed representation of a recent log
  """
    comment = _FormatIssueBody(info, log_data)
    url = _FormatNewIssueUrl(comment.body)
    if len(url) > MAX_URL_LENGTH:
        max_info_len = MAX_URL_LENGTH - len(_FormatNewIssueUrl(''))
        truncated, remaining = _ShortenIssueBody(comment, max_info_len)
        log.warning('Truncating included information. Please consider including the remainder:')
        divider_text = 'TRUNCATED INFORMATION (PLEASE CONSIDER INCLUDING)'
        log.status.Print(GetDivider(divider_text))
        log.status.Print(remaining.strip())
        log.status.Print(GetDivider('END ' + divider_text))
        log.warning('The output of gcloud info is too long to pre-populate the new issue form.')
        log.warning('Please consider including the remainder (above).')
        url = _FormatNewIssueUrl(truncated)
    OpenInBrowser(url)
    log.status.Print('Opening your browser to a new Google Cloud SDK issue.')
    log.status.Print('If your browser does not open or you have issues loading the web page, please ensure you are signed into your account on %s first, then try again.' % ISSUE_TRACKER_BASE_URL)
    log.status.Print('If you still have issues loading the web page, please file an issue: %s' % ISSUE_TRACKER_URL)