import faulthandler
import json
import logging
import os
import time
import traceback
import warnings
from typing import Any, Dict, Optional
def _write_error_file(self, file_path: str, error_msg: str) -> None:
    """Write error message to the file."""
    try:
        with open(file_path, 'w') as fp:
            fp.write(error_msg)
    except Exception as e:
        warnings.warn(f'Unable to write error to file. {type(e).__name__}: {e}')