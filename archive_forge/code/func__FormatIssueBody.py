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
def _FormatIssueBody(info, log_data=None):
    """Construct a useful issue body with which to pre-populate the issue tracker.

  Args:
    info: InfoHolder, holds information about the Cloud SDK install
    log_data: LogData, parsed log data for a gcloud run

  Returns:
    CommentHolder, a class containing the issue comment body, part of comment
        before stacktrace, the stacktrace portion of the comment, and the
        exception
  """
    gcloud_info = six.text_type(info)
    formatted_command = ''
    if log_data and log_data.command:
        formatted_command = 'Issue running command [{0}].\n\n'.format(log_data.command)
    pre_stacktrace = COMMENT_PRE_STACKTRACE_TEMPLATE.format(formatted_command=formatted_command)
    formatted_traceback = ''
    formatted_stacktrace = ''
    exception = ''
    if log_data and log_data.traceback:
        formatted_stacktrace, exception = _FormatTraceback(log_data.traceback)
        formatted_traceback = 'Trace:\n' + formatted_stacktrace + exception
    comment_body = COMMENT_TEMPLATE.format(formatted_command=formatted_command, gcloud_info=gcloud_info.strip(), formatted_traceback=formatted_traceback)
    return CommentHolder(comment_body, pre_stacktrace, formatted_stacktrace, exception)