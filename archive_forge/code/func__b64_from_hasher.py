import base64
import hashlib
import mmap
import os
import sys
from pathlib import Path
from typing import NewType, Union
from wandb.sdk.lib.paths import StrPath
def _b64_from_hasher(hasher: 'hashlib._Hash') -> B64MD5:
    return B64MD5(base64.b64encode(hasher.digest()).decode('ascii'))