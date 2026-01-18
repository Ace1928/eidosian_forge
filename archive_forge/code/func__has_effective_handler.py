import logging
import sys
from typing import Union
from rq.defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT
def _has_effective_handler(logger) -> bool:
    """
    Checks if a logger has a handler that will catch its messages in its logger hierarchy.

    Args:
        logger (logging.Logger): The logger to be checked.

    Returns:
        is_configured (bool): True if a handler is found for the logger, False otherwise.
    """
    while True:
        if logger.handlers:
            return True
        if not logger.parent:
            return False
        logger = logger.parent