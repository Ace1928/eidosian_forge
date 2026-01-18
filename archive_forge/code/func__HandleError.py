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
def _HandleError(self, e):
    message = bq_logging.EncodeForPrinting(e)
    if isinstance(e, bq_error.BigqueryClientConfigurationError):
        message += ' Try running "bq init".'
    print('Exception raised in %s operation: %s' % (self._command_name, message))
    return 1