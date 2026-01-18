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
def _PrettyArgs(args):
    """Pretty prints args into a string and returns it."""
    buf = io.StringIO()
    buf.write('[')
    for arg in args:
        buf.write('({},{})'.format(arg.value or '""', arg.token_type.name))
    buf.write(']')
    return buf.getvalue()