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
def More(contents, out=None, prompt=None, check_pager=True):
    """Run a user specified pager or fall back to the internal pager.

  Args:
    contents: The entire contents of the text lines to page.
    out: The output stream, log.out (effectively) if None.
    prompt: The page break prompt.
    check_pager: Checks the PAGER env var and uses it if True.
  """
    if not IsInteractive(output=True):
        if not out:
            out = log.out
        out.write(contents)
        return
    if not out:
        log.file_only_logger.info(contents)
        out = sys.stdout
    if check_pager:
        pager = encoding.GetEncodedValue(os.environ, 'PAGER', None)
        if pager == '-':
            pager = None
        elif not pager:
            for command in ('less', 'pager'):
                if files.FindExecutableOnPath(command):
                    pager = command
                    break
        if pager:
            less_orig = encoding.GetEncodedValue(os.environ, 'LESS', None)
            less = '-R' + (less_orig or '')
            encoding.SetEncodedValue(os.environ, 'LESS', less)
            p = subprocess.Popen(pager, stdin=subprocess.PIPE, shell=True)
            enc = console_attr.GetConsoleAttr().GetEncoding()
            p.communicate(input=contents.encode(enc))
            p.wait()
            if less_orig is None:
                encoding.SetEncodedValue(os.environ, 'LESS', None)
            return
    console_pager.Pager(contents, out, prompt).Run()