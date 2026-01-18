import logging
import os
from logging import (
from typing import Optional
def _configure_library_root_logger() -> None:
    library_root_logger = _get_library_root_logger()
    library_root_logger.addHandler(logging.StreamHandler())
    library_root_logger.setLevel(_get_default_logging_level())