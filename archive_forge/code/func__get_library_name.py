import logging
import os
from logging import (
from typing import Optional
def _get_library_name() -> str:
    return __name__.split('.')[0]