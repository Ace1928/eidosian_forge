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
def _DeduceBetterError(self, context, args, namespace):
    """There is an argparse error in context, see if we can do better.

    We are committed to an argparse error. See if we can do better than the
    observed error in context by isolating each flag arg to determine if the
    argparse error complained about a flag arg value instead of a positional.

    Args:
      context: The _ErrorContext containing the error to improve.
      args: The subset of the command lines args that triggered the argparse
        error in context.
      namespace: The namespace for the current parser.
    """
    self._probe_error = True
    for arg in args:
        try:
            if not arg.startswith('-'):
                break
        except AttributeError:
            break
        _, _, error_context = self._ParseKnownArgs([arg], namespace)
        if not error_context:
            continue
        context = error_context
        break
    self._probe_error = False
    context.error.argument = context.AddLocations(context.error.argument)
    context.parser.error(context=context, reproduce=True)