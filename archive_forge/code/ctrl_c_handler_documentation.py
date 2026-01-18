from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import signal
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
Cancel a test matrix if CTRL-C is typed during a section of code.

  While within this context manager, the CTRL-C signal is caught and a test
  matrix is cancelled. This should only be used with a section of code where
  the test matrix is running.
  