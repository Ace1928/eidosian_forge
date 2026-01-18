import contextlib
import os
import shutil
import stat
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Generator, Optional, Union
import yaml
@contextlib.contextmanager
def SoftTemporaryDirectory(suffix: Optional[str]=None, prefix: Optional[str]=None, dir: Optional[Union[Path, str]]=None, **kwargs) -> Generator[str, None, None]:
    """
    Context manager to create a temporary directory and safely delete it.

    If tmp directory cannot be deleted normally, we set the WRITE permission and retry.
    If cleanup still fails, we give up but don't raise an exception. This is equivalent
    to  `tempfile.TemporaryDirectory(..., ignore_cleanup_errors=True)` introduced in
    Python 3.10.

    See https://www.scivision.dev/python-tempfile-permission-error-windows/.
    """
    tmpdir = tempfile.TemporaryDirectory(prefix=prefix, suffix=suffix, dir=dir, **kwargs)
    yield tmpdir.name
    try:
        shutil.rmtree(tmpdir.name)
    except Exception:
        try:
            shutil.rmtree(tmpdir.name, onerror=_set_write_permission_and_retry)
        except Exception:
            pass
    try:
        tmpdir.cleanup()
    except Exception:
        pass