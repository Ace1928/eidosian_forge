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
class _HandleLaterError(Exception):
    """Error to be handled in a subsequent call to self.error.

  This error exists to provide a way to break out of self.error so that we can
  deduce a better error later; it will always be caught in parser_extensions and
  never surfaced as a user-facing error (at least in theory; if that does happen
  then it's a bug.)
  """
    pass