from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import argparse
import collections
import io
import itertools
import os
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base  # pylint: disable=unused-import
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import suggest_commands
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
import six
def ReportErrorMetrics(self, error, message):
    """Reports Command and Error metrics in case of argparse errors.

    Args:
      error: Exception, The Exception object.
      message: str, The exception error message.
    """
    dotted_command_path = '.'.join(self._calliope_command.GetPath())
    if isinstance(error, parser_errors.ArgumentError):
        if error.extra_path_arg:
            dotted_command_path = '.'.join([dotted_command_path, error.extra_path_arg])
        self._ReportErrorMetricsHelper(dotted_command_path, error.__class__, error.error_extra_info)
        return
    if 'too few arguments' in message:
        self._ReportErrorMetricsHelper(dotted_command_path, parser_errors.TooFewArgumentsError)
        return
    self._ReportErrorMetricsHelper(dotted_command_path, parser_errors.OtherParsingError)