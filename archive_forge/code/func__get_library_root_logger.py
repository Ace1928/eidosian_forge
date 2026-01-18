import logging
import os
from logging import (
from typing import Optional
def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())