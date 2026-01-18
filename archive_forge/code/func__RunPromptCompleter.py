from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import signal
import sys
import time
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
def _RunPromptCompleter(self, args):
    completer_class = module_util.ImportModule(args.prompt_completer)
    choices = parser_completer.ArgumentCompleter(completer_class, args)
    response = console_io.PromptResponse('Complete this: ', choices=choices)
    print(response)