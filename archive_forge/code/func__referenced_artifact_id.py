import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union
from urllib.parse import urlparse
from wandb.errors.term import termwarn
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.hashutil import (
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath, URIStr
def _referenced_artifact_id(self) -> Optional[str]:
    if not self._is_artifact_reference():
        return None
    return hex_to_b64_id(urlparse(self.ref).netloc)