from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import os
import re
import sys
import types
import uuid
import argcomplete
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import backend
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
def _ApplyFlagsFile(args):
    """Applies FLAGS_FILE_FLAG in args and returns the new args.

  The basic algorithm is arg list manipulation, done before ArgParse is called.
  This function reaps all FLAGS_FILE_FLAG args from the command line, and
  recursively from the flags files, and inserts them into a new args list by
  replacing the --flags-file=YAML-FILE flag by its constituent flags. This
  preserves the left-to-right precedence of the argument parser. Internal
  _FLAG_FILE_LINE_NAME flags are also inserted into args. This specifies the
  flags source file and line number for each flags file flag, and is used to
  construct actionable error messages.

  Args:
    args: The original args list.

  Returns:
    A new args list with all FLAGS_FILE_FLAG args replaced by their constituent
    flags.
  """
    flag = calliope_base.FLAGS_FILE_FLAG.name
    flag_eq = flag + '='
    if not any([arg == flag or arg.startswith(flag_eq) for arg in args]):
        return args
    peek = False
    new_args = []
    for arg in args:
        if peek:
            peek = False
            _AddFlagsFileFlags(new_args, arg)
        elif arg == flag:
            peek = True
        elif arg.startswith(flag_eq):
            _AddFlagsFileFlags(new_args, arg[len(flag_eq):])
        else:
            new_args.append(arg)
    return new_args