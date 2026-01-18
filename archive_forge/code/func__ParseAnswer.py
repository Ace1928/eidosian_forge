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
def _ParseAnswer(answer, options, allow_freeform):
    """Parses answer and returns 1-based index in options list.

  Args:
    answer: str, The answer input by the user to be parsed as a choice.
    options: [object], A list of objects to select.  Their str()
          method will be used to select them via freeform text.
    allow_freeform: bool, A flag which, if defined, will allow the user to input
          the choice as a str, not just as a number. If not set, only numbers
          will be accepted.

  Returns:
    int, The 1-indexed value in the options list that corresponds to the answer
          that was given, or None if the selection is invalid. Note that this
          function does not do any validation that the value is a valid index
          (in the case that an integer answer was given)
  """
    try:
        return int(answer)
    except ValueError:
        pass
    if not allow_freeform:
        return None
    try:
        return list(map(str, options)).index(answer) + 1
    except ValueError:
        pass
    return None