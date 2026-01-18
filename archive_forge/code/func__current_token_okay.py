import os
import subprocess
from functools import partial
from getpass import getpass
from pathlib import Path
from typing import Optional
from . import constants
from .commands._cli_utils import ANSI
from .utils import (
from .utils._token import _get_token_from_environment, _get_token_from_google_colab
def _current_token_okay(write_permission: bool=False):
    """Check if the current token is valid.

    Args:
        write_permission (`bool`, defaults to `False`):
            If `True`, requires a token with write permission.

    Returns:
        `bool`: `True` if the current token is valid, `False` otherwise.
    """
    from .hf_api import get_token_permission
    permission = get_token_permission()
    if permission is None or (write_permission and permission != 'write'):
        return False
    return True