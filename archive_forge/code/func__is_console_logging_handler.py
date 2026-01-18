import logging
import sys
from contextlib import contextmanager
from ..std import tqdm as std_tqdm
def _is_console_logging_handler(handler):
    return isinstance(handler, logging.StreamHandler) and handler.stream in {sys.stdout, sys.stderr}