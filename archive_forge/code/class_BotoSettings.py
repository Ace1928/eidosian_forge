import os
import json
import pathlib
from typing import Optional, Union, Dict, Any
from lazyops.types.models import BaseSettings, validator
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
class BotoSettings(BaseSettings):
    boto_config: Optional[Union[str, pathlib.Path]] = None
    boto_path: Optional[Union[str, pathlib.Path]] = None

    @lazyproperty
    def path(self) -> FileLike:
        p = self.boto_config or self.boto_path
        if p is None:
            return None
        if _fileio_available:
            return File(p)
        if isinstance(p, str):
            p = pathlib.Path(p)
        return p

    @lazyproperty
    def exists(self) -> bool:
        return False if self.path is None else self.path.exists()

    def set_env(self):
        """
        Update the Env variables for the current session
        """
        if self.exists:
            os.environ['BOTO_CONFIG'] = self.path.as_posix()
            os.environ['BOTO_PATH'] = self.path.as_posix()