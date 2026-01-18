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
def DefaultStreamStructuredOutHandler(result_holder, capture_output=False, warn_if_not_stuctured=True):
    """Default processing for structured stdout from threaded subprocess."""

    def HandleStdOut(line):
        """Process structured stdout."""
        if line:
            msg_rec = line.strip()
            try:
                msg, resources = _LogStructuredStdOut(msg_rec)
                if capture_output:
                    _CaptureStdOut(result_holder, output_message=msg, resource_output=resources)
            except StructuredOutputError as sme:
                if warn_if_not_stuctured:
                    log.warning(_STRUCTURED_TEXT_EXPECTED_ERROR.format(sme))
                log.out.Print(msg_rec)
                _CaptureStdOut(result_holder, raw_output=msg_rec)
    return HandleStdOut