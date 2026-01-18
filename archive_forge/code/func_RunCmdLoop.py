import logging
import os
import pdb
import shlex
import sys
import traceback
import types
from absl import app
from absl import flags
import googleapiclient
import bq_flags
import bq_utils
from utils import bq_error
from utils import bq_logging
from utils import bq_processor_utils
from pyglib import appcommands
def RunCmdLoop(self, argv):
    """Hook for use in cmd.Cmd-based command shells."""
    try:
        args = shlex.split(argv)
    except ValueError as e:
        raise SyntaxError(bq_logging.EncodeForPrinting(e)) from e
    return self.Run([self._command_name] + args)