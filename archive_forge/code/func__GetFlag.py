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
def _GetFlag(self, flagname):
    if flagname in self._command_flags:
        return self._command_flags[flagname]
    else:
        return None