from pyarrow.util import _is_path_like, _stringify_path
from pyarrow._fs import (  # noqa
def _delete_dir_contents(self, path, missing_dir_ok):
    try:
        subpaths = self.fs.listdir(path, detail=False)
    except FileNotFoundError:
        if missing_dir_ok:
            return
        raise
    for subpath in subpaths:
        if self.fs.isdir(subpath):
            self.fs.rm(subpath, recursive=True)
        elif self.fs.isfile(subpath):
            self.fs.rm(subpath)