from pyarrow.util import _is_path_like, _stringify_path
from pyarrow._fs import (  # noqa
@staticmethod
def _create_file_info(path, info):
    size = info['size']
    if info['type'] == 'file':
        ftype = FileType.File
    elif info['type'] == 'directory':
        ftype = FileType.Directory
        size = None
    else:
        ftype = FileType.Unknown
    return FileInfo(path, ftype, size=size, mtime=info.get('mtime', None))