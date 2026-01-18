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
class _ErrorContext(object):
    """Context from the most recent ArgumentParser.error() call.

  The context can be saved and used to reproduce the error() method call later
  in the execution.  Used to probe argparse errors for different argument
  combinations.

  Attributes:
    message: The error message string.
    parser: The parser where the error occurred.
    error: The exception error value.
  """

    def __init__(self, message, parser, error):
        self.message = re.sub("\\bu'", "'", message)
        self.parser = parser
        self.error = error
        self.flags_locations = parser.flags_locations

    def AddLocations(self, arg):
        """Adds locaton info from context for arg if specified."""
        locations = self.flags_locations.get(arg)
        if locations:
            arg = '{} ({})'.format(arg, ','.join(sorted(locations)))
        return arg