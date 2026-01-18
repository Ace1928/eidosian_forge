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
def _IsSingleCharFlag(argument):
    """Determines if the argument is a single char flag (e.g. '-a')."""
    return re.match('^-[a-zA-Z]$', argument) or re.match('^-[a-zA-Z]=', argument)