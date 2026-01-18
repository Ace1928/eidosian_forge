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
class FireError(Exception):
    """Exception used by Fire when a Fire command cannot be executed.

  These exceptions are not raised by the Fire function, but rather are caught
  and added to the FireTrace.
  """