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
def _GenerateCompletions(event):
    """completion.generate_completions override that auto selects singletons."""
    b = event.current_buffer
    if not b.complete_state:
        event.cli.start_completion(insert_common_part=True, select_first=False)
    elif len(b.complete_state.current_completions) == 1:
        b.apply_completion(b.complete_state.current_completions[0])
    else:
        b.complete_next()