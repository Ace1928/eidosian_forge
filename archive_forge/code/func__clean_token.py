import os
import warnings
from pathlib import Path
from threading import Lock
from typing import Optional
from .. import constants
from ._runtime import is_google_colab
def _clean_token(token: Optional[str]) -> Optional[str]:
    """Clean token by removing trailing and leading spaces and newlines.

    If token is an empty string, return None.
    """
    if token is None:
        return None
    return token.replace('\r', '').replace('\n', '').strip() or None