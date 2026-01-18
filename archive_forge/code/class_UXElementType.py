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
class UXElementType(enum.Enum):
    """Describes the type of a ux element."""
    PROGRESS_BAR = (0, 'message')
    PROGRESS_TRACKER = (1, 'message', 'aborted_message', 'status')
    STAGED_PROGRESS_TRACKER = (2, 'message', 'status', 'succeeded_stages', 'failed_stage')
    PROMPT_CONTINUE = (3, 'message', 'prompt_string', 'cancel_string')
    PROMPT_RESPONSE = (4, 'message')
    PROMPT_CHOICE = (5, 'message', 'prompt_string', 'choices')
    PROMPT_WITH_VALIDATOR = (6, 'error_message', 'prompt_string', 'message', 'allow_invalid')

    def __init__(self, ordinal, *data_fields):
        del ordinal
        self._data_fields = data_fields

    def GetDataFields(self):
        """Returns the ordered list of additional fields in the UX Element."""
        return self._data_fields