from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from googlecloudsdk.command_lib.util.anthos import structured_messages as sm
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def HandleStdErr(line):
    """Handle line as a structured message.

    Attempts to parse line into an OutputMessage and log any errors or warnings
    accordingly. If line cannot be parsed as an OutputMessage, logs it as plain
    text to stderr. If capture_output is True will capture any logged text to
    result_holder.

    Args:
      line: string, line of output read from stderr.
    """
    if line:
        msg_rec = line.strip()
        try:
            msg = ReadStructuredOutput(line)
            if msg.IsError():
                if msg.level == 'info':
                    log.info(msg.error_details.Format())
                elif msg.level == 'error':
                    log.error(msg.error_details.Format())
                elif msg.level == 'warning':
                    log.warning(msg.error_details.Format())
                elif msg.level == 'debug':
                    log.debug(msg.error_details.Format())
            else:
                log.status.Print(msg.body)
            if capture_output:
                _CaptureStdErr(result_holder, output_message=msg)
        except StructuredOutputError as sme:
            if warn_if_not_stuctured:
                log.warning(_STRUCTURED_TEXT_EXPECTED_ERROR.format(sme))
            log.status.Print(msg_rec)
            if capture_output:
                _CaptureStdErr(result_holder, raw_output=msg_rec)