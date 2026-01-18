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
def _ParseKnownArgs(self, args, namespace, wrapper=True):
    """Calls parse_known_args() and adds error_context to the return.

    Args:
      args: The list of command line args.
      namespace: The parsed args namespace.
      wrapper: Calls the parse_known_args() wrapper if True, otherwise the
        wrapped argparse parse_known_args().

    Returns:
      namespace: The parsed arg namespace.
      unknown_args: The list of unknown args.
      error_context: The _ErrorContext if there was an error, None otherwise.
    """
    self._error_context = None
    parser = self if wrapper else super(ArgumentParser, self)
    try:
        namespace, unknown_args = parser.parse_known_args(args, namespace)
    except _HandleLaterError:
        unknown_args = []
    error_context = self._error_context
    self._error_context = None
    if not unknown_args and hasattr(parser, 'flags_locations'):
        parser.flags_locations = collections.defaultdict(set)
    return (namespace, unknown_args, error_context)