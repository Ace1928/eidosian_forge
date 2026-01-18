from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import sys
import threading
import time
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.command_lib.meta import generate_cli_trees
from googlecloudsdk.core import module_util
from googlecloudsdk.core.console import console_attr
from prompt_toolkit import completion
import six
Returns the Completion object for a file/uri path completion value.

    Args:
      value: The file/path completion value string.
      offset: The Completion object offset used for dropdown display.
      chop: The minimum number of chars to chop from the dropdown items.
      strip_trailing_slash: Strip trailing '/' if True.

    Returns:
      The Completion object for a file path completion value or None if the
      chopped/stripped value is empty.
    