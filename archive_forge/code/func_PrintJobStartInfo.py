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
def PrintJobStartInfo(self, job):
    """Print a simple status line."""
    if FLAGS.format in ['prettyjson', 'json']:
        bq_utils.PrintFormattedJsonObject(job)
    else:
        reference = bq_processor_utils.ConstructObjectReference(job)
        print('Successfully started %s %s' % (self._command_name, reference))