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
def _SuggestFreeformAnswer(suggester, answer, options):
    """Checks if there is a suitable close choice to suggest.

  Args:
    suggester: object, An object which has methods AddChoices and
      GetSuggestion which is used to detect if an answer which is not present
      in the options list is a likely typo, and to provide a suggestion
      accordingly.
    answer: str, The freeform answer input by the user as a choice.
    options: [object], A list of objects to select.  Their str()
          method will be used to compare them to answer.

  Returns:
    str, the closest option in options to answer, or None otherwise.
  """
    suggester.AddChoices(map(str, options))
    return suggester.GetSuggestion(answer)