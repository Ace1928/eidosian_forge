import warnings
from pathlib import Path
from typing import Optional
from .. import constants
from ._token import get_token

        Deletes the token from storage. Does not fail if token does not exist.
        