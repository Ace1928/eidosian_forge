import faulthandler
import json
import logging
import os
import time
import traceback
import warnings
from typing import Any, Dict, Optional
def _get_error_file_path(self) -> Optional[str]:
    """
        Return the error file path.

        May return ``None`` to have the structured error be logged only.
        """
    return os.environ.get('TORCHELASTIC_ERROR_FILE', None)