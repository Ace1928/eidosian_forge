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
def GetSpecifiedArgs(self):
    """Gets the argument names and values that were actually specified.

    For example,

      `$ {command} positional_value --foo=bar, --lorem-ipsum=1,2,3`

    returns
      {
        'POSITIONAL_NAME': 'positional_value'
        '--foo': 'bar',
        '--lorem-ipsum': [1,2,3],
      }

    In the returned dictionary, the keys are the specified arguments, including
    positional argument names and flag names, in string type; the corresponding
    values are the user-specified flag values, converted according to the type
    defined by each flag.

    Returns:
      {str: any}, A mapping of argument name to value.
    """
    return {name: getattr(self, dest, 'UNKNOWN') for dest, name in six.iteritems(self._specified_args)}