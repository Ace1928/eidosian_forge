import base64
import hashlib
import mmap
import os
import sys
from pathlib import Path
from typing import NewType, Union
from wandb.sdk.lib.paths import StrPath
def _md5_file_hasher(*paths: StrPath) -> 'hashlib._Hash':
    md5_hash = _md5()
    for path in sorted((Path(p) for p in paths)):
        with path.open('rb') as f:
            if os.stat(f.fileno()).st_size <= 1024 * 1024:
                md5_hash.update(f.read())
            else:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mview:
                    md5_hash.update(mview)
    return md5_hash