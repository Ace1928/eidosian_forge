import logging
import sys
from contextlib import contextmanager
from ..std import tqdm as std_tqdm
def _get_first_found_console_logging_handler(handlers):
    for handler in handlers:
        if _is_console_logging_handler(handler):
            return handler