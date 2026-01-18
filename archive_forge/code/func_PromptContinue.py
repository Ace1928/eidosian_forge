from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import enum
import getpass
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_pager
from googlecloudsdk.core.console import prompt_completer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def PromptContinue(message=None, prompt_string=None, default=True, throw_if_unattended=False, cancel_on_no=False, cancel_string=None):
    """Prompts the user a yes or no question and asks if they want to continue.

  Args:
    message: str, The prompt to print before the question.
    prompt_string: str, An alternate yes/no prompt to display.  If None, it
      defaults to 'Do you want to continue'.
    default: bool, What the default answer should be.  True for yes, False for
      no.
    throw_if_unattended: bool, If True, this will throw if there was nothing
      to consume on stdin and stdin is not a tty.
    cancel_on_no: bool, If True and the user answers no, throw an exception to
      cancel the entire operation.  Useful if you know you don't want to
      continue doing anything and don't want to have to raise your own
      exception.
    cancel_string: str, An alternate error to display on No. If None, it
      defaults to 'Aborted by user.'.

  Raises:
    UnattendedPromptError: If there is no input to consume and this is not
      running in an interactive terminal.
    OperationCancelledError: If the user answers no and cancel_on_no is True.

  Returns:
    bool, False if the user said no, True if the user said anything else or if
    prompts are disabled.
  """
    if properties.VALUES.core.disable_prompts.GetBool():
        if not default and cancel_on_no:
            raise OperationCancelledError()
        return default
    style = properties.VALUES.core.interactive_ux_style.Get()
    prompt_generator = _TestPromptContinuePromptGenerator if style == properties.VALUES.core.InteractiveUXStyles.TESTING.name else _NormalPromptContinuePromptGenerator
    prompt, reprompt, ending = prompt_generator(message=message, prompt_string=prompt_string, default=default, throw_if_unattended=throw_if_unattended, cancel_on_no=cancel_on_no, cancel_string=cancel_string)
    sys.stderr.write(prompt)

    def GetAnswer(reprompt):
        """Get answer to input prompt."""
        while True:
            answer = _GetInput()
            if answer == '':
                return default
            elif answer is None:
                if throw_if_unattended and (not IsInteractive()):
                    raise UnattendedPromptError()
                else:
                    return default
            elif answer.strip().lower() in ['y', 'yes']:
                return True
            elif answer.strip().lower() in ['n', 'no']:
                return False
            elif reprompt:
                sys.stderr.write(reprompt)
    try:
        answer = GetAnswer(reprompt)
    finally:
        if ending:
            sys.stderr.write(ending)
    if not answer and cancel_on_no:
        raise OperationCancelledError(cancel_string)
    return answer