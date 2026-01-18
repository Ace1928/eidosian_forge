from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import json
import os
import pipes
import re
import shlex
import sys
import types
from fire import completion
from fire import decorators
from fire import formatting
from fire import helptext
from fire import inspectutils
from fire import interact
from fire import parser
from fire import trace
from fire import value_types
from fire.console import console_io
import six
class FireExit(SystemExit):
    """An exception raised by Fire to the client in the case of a FireError.

  The trace of the Fire program is available on the `trace` property.

  This exception inherits from SystemExit, so clients may explicitly catch it
  with `except SystemExit` or `except FireExit`. If not caught, this exception
  will cause the client program to exit without a stacktrace.
  """

    def __init__(self, code, component_trace):
        """Constructs a FireExit exception.

    Args:
      code: (int) Exit code for the Fire CLI.
      component_trace: (FireTrace) The trace for the Fire command.
    """
        super(FireExit, self).__init__(code)
        self.trace = component_trace