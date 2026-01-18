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
def GetSpecifiedArgsDict(self):
    """Returns the _specified_args dictionary.

    For example,

      $ {command} positional_value --foo=bar, --lorem-ipsum=hello --async,

    returns
      {
        'positional_name', 'POSITIONAL_NAME'
        'foo': '--foo',
        'lorem_ipsum': '--lorem-ipsum',
        'async_': '--async',
      }.

    In the returned dictionary, the keys are destinations in the argparse
    namespace object.

    In the above example, the destination of `--async` is set to 'async_' in its
    flag definition, other flags use underscore separated flag names as their
    default destinations.
    """
    return self._specified_args