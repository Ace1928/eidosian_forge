from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import errno
import os
import pdb
import sys
import textwrap
import traceback
from absl import command_name
from absl import flags
from absl import logging
def _call_exception_handlers(exception):
    """Calls any installed exception handlers."""
    for handler in EXCEPTION_HANDLERS:
        try:
            if handler.wants(exception):
                handler.handle(exception)
        except:
            try:
                logging.error(traceback.format_exc())
            except:
                pass