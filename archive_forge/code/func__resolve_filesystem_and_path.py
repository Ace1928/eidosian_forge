from pyarrow.util import _is_path_like, _stringify_path
from pyarrow._fs import (  # noqa
def _resolve_filesystem_and_path(path, filesystem=None, allow_legacy_filesystem=False, memory_map=False):
    """
    Return filesystem/path from path which could be an URI or a plain
    filesystem path.
    """
    if not _is_path_like(path):
        if filesystem is not None:
            raise ValueError("'filesystem' passed but the specified path is file-like, so there is nothing to open with 'filesystem'.")
        return (filesystem, path)
    if filesystem is not None:
        filesystem = _ensure_filesystem(filesystem, use_mmap=memory_map, allow_legacy_filesystem=allow_legacy_filesystem)
        if isinstance(filesystem, LocalFileSystem):
            path = _stringify_path(path)
        elif not isinstance(path, str):
            raise TypeError('Expected string path; path-like objects are only allowed with a local filesystem')
        if not allow_legacy_filesystem:
            path = filesystem.normalize_path(path)
        return (filesystem, path)
    path = _stringify_path(path)
    filesystem = LocalFileSystem(use_mmap=memory_map)
    try:
        file_info = filesystem.get_file_info(path)
    except ValueError:
        file_info = None
        exists_locally = False
    else:
        exists_locally = file_info.type != FileType.NotFound
    if not exists_locally:
        try:
            filesystem, path = FileSystem.from_uri(path)
        except ValueError as e:
            if 'empty scheme' not in str(e) and 'Cannot parse URI' not in str(e):
                raise
    else:
        path = filesystem.normalize_path(path)
    return (filesystem, path)