from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def _ConfirmWithUserPrompt(question, default_response):
    """Prompts user to confirm an action with yes or no response.

  Args:
    question: Yes/No question to be used for the prompt.
    default_response: Default response to the question: True, False

  Returns:
    Returns the rough equivalent duration in seconds.
  """
    prompt = ''
    if default_response:
        prompt = '%s [%s|%s]: ' % (question, 'Y', 'n')
    else:
        prompt = '%s [%s|%s]: ' % (question, 'y', 'N')
    while True:
        response = input(prompt).lower()
        if not response:
            return default_response
        if response not in ['y', 'yes', 'n', 'no']:
            print("\tPlease respond with 'yes'/'y' or 'no'/'n'.")
            continue
        if response == 'yes' or response == 'y':
            return True
        if response == 'no' or response == 'n':
            return False