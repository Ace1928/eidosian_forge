import logging
import sys
from typing import Optional, TextIO
from absl import flags
from absl import logging as absl_logging
from googleapiclient import model
def _SetLogFile(logfile: TextIO):
    absl_logging.use_python_logging(quiet=True)
    absl_logging.get_absl_handler().python_handler.stream = logfile