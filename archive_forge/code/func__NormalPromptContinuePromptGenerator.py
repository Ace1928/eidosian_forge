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
def _NormalPromptContinuePromptGenerator(message, prompt_string, default, throw_if_unattended, cancel_on_no, cancel_string):
    """Generates prompts for prompt continue under normal conditions."""
    del throw_if_unattended
    del cancel_on_no
    del cancel_string
    buf = io.StringIO()
    if message:
        buf.write(_DoWrap(message) + '\n\n')
    if not prompt_string:
        prompt_string = 'Do you want to continue'
    if default:
        prompt_string += ' (Y/n)?  '
    else:
        prompt_string += ' (y/N)?  '
    buf.write(_DoWrap(prompt_string))
    return (buf.getvalue(), "Please enter 'y' or 'n':  ", '\n')