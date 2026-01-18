from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib import feedback_util
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import text as text_util
import six
from six.moves import map
def _SuggestIncludeRecentLogs():
    recent_runs = info_holder.LogsInfo().GetRecentRuns()
    if recent_runs:
        now = datetime.datetime.now()

        def _FormatLogData(run):
            crash = ' (crash detected)' if run.traceback else ''
            time = 'Unknown time'
            if run.date:
                time = text_util.PrettyTimeDelta(now - run.date) + ' ago'
            return '[{0}]{1}: {2}'.format(run.command, crash, time)
        idx = console_io.PromptChoice(list(map(_FormatLogData, recent_runs)) + ['None of these'], default=0, message='Which recent gcloud invocation would you like to provide feedback about? This will open a new browser tab.')
        if idx < len(recent_runs):
            return recent_runs[idx]