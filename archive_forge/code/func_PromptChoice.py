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
def PromptChoice(options, default=None, message=None, prompt_string=None, allow_freeform=False, freeform_suggester=None, cancel_option=False):
    """Prompt the user to select a choice from a list of items.

  Args:
    options:  [object], A list of objects to print as choices.  Their
      six.text_type() method will be used to display them.
    default: int, The default index to return if prompting is disabled or if
      they do not enter a choice.
    message: str, An optional message to print before the choices are displayed.
    prompt_string: str, A string to print when prompting the user to enter a
      choice.  If not given, a default prompt is used.
    allow_freeform: bool, A flag which, if defined, will allow the user to input
      the choice as a str, not just as a number. If not set, only numbers will
      be accepted.
    freeform_suggester: object, An object which has methods AddChoices and
      GetSuggestion which is used to detect if an answer which is not present
      in the options list is a likely typo, and to provide a suggestion
      accordingly.
    cancel_option: bool, A flag indicating whether an option to cancel the
      operation should be added to the end of the list of choices.

  Raises:
    ValueError: If no options are given or if the default is not in the range of
      available options.
    OperationCancelledError: If a `cancel` option is selected by user.

  Returns:
    The index of the item in the list that was chosen, or the default if prompts
    are disabled.
  """
    if not options:
        raise ValueError('You must provide at least one option.')
    options = options + ['cancel'] if cancel_option else options
    maximum = len(options)
    if default is not None and (not 0 <= default < maximum):
        raise ValueError('Default option [{default}] is not a valid index for the options list [{maximum} options given]'.format(default=default, maximum=maximum))
    if properties.VALUES.core.disable_prompts.GetBool():
        return default
    style = properties.VALUES.core.interactive_ux_style.Get()
    if style == properties.VALUES.core.InteractiveUXStyles.TESTING.name:
        write = lambda x: None
        sys.stderr.write(JsonUXStub(UXElementType.PROMPT_CHOICE, message=message, prompt_string=prompt_string, choices=[six.text_type(o) for o in options]) + '\n')
    else:
        write = sys.stderr.write
    if message:
        write(_DoWrap(message) + '\n')
    if maximum > PROMPT_OPTIONS_OVERFLOW:
        _PrintOptions(options, write, limit=PROMPT_OPTIONS_OVERFLOW)
        truncated = maximum - PROMPT_OPTIONS_OVERFLOW
        write('Did not print [{truncated}] options.\n'.format(truncated=truncated))
        write('Too many options [{maximum}]. Enter "list" at prompt to print choices fully.\n'.format(maximum=maximum))
    else:
        _PrintOptions(options, write)
    if not prompt_string:
        if allow_freeform:
            prompt_string = 'Please enter numeric choice or text value (must exactly match list item)'
        else:
            prompt_string = 'Please enter your numeric choice'
    if default is None:
        suffix_string = ':  '
    else:
        suffix_string = ' ({default}):  '.format(default=default + 1)

    def _PrintPrompt():
        write(_DoWrap(prompt_string + suffix_string))
    _PrintPrompt()
    while True:
        answer = _GetInput()
        if answer is None or (not answer and default is not None):
            write('\n')
            if cancel_option and default == maximum - 1:
                raise OperationCancelledError()
            return default
        if answer == 'list':
            _PrintOptions(options, write)
            _PrintPrompt()
            continue
        num_choice = _ParseAnswer(answer, options, allow_freeform)
        if cancel_option and num_choice == maximum:
            raise OperationCancelledError()
        if num_choice is not None and num_choice >= 1 and (num_choice <= maximum):
            write('\n')
            return num_choice - 1
        if allow_freeform and freeform_suggester:
            suggestion = _SuggestFreeformAnswer(freeform_suggester, answer, options)
            if suggestion is not None:
                write('[{answer}] not in list. Did you mean [{suggestion}]?'.format(answer=answer, suggestion=suggestion))
                write('\n')
        if allow_freeform:
            write('Please enter a value between 1 and {maximum}, or a value present in the list:  '.format(maximum=maximum))
        else:
            write('Please enter a value between 1 and {maximum}:  '.format(maximum=maximum))