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
class _StubProgressBar(object):
    """A progress bar that only prints deterministic start and end points.

  No UX about progress should be exposed here. This is strictly for being able
  to tell that the progress bar was invoked, not what it actually looks like.
  """

    def __init__(self, label, stream):
        self._raw_label = label
        self._stream = stream

    def Start(self):
        self._stream.write(JsonUXStub(UXElementType.PROGRESS_BAR, message=self._raw_label))

    def SetProgress(self, progress_factor):
        pass

    def Finish(self):
        """Mark the progress as done."""
        self.SetProgress(1)
        self._stream.write('\n')

    def __enter__(self):
        self.Start()
        return self

    def __exit__(self, *args):
        self.Finish()