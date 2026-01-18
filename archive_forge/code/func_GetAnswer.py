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