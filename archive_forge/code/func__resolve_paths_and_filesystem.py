import pathlib
import sys
import urllib
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from ray.data._internal.util import _resolve_custom_scheme
def _resolve_paths_and_filesystem(paths: Union[str, List[str]], filesystem: 'pyarrow.fs.FileSystem'=None) -> Tuple[List[str], 'pyarrow.fs.FileSystem']:
    """
    Resolves and normalizes all provided paths, infers a filesystem from the
    paths and ensures that all paths use the same filesystem.

    Args:
        paths: A single file/directory path or a list of file/directory paths.
            A list of paths can contain both files and directories.
        filesystem: The filesystem implementation that should be used for
            reading these files. If None, a filesystem will be inferred. If not
            None, the provided filesystem will still be validated against all
            filesystems inferred from the provided paths to ensure
            compatibility.
    """
    import pyarrow as pa
    from pyarrow.fs import FileSystem, FSSpecHandler, PyFileSystem, _resolve_filesystem_and_path
    if isinstance(paths, str):
        paths = [paths]
    if isinstance(paths, pathlib.Path):
        paths = [str(paths)]
    elif not isinstance(paths, list) or any((not isinstance(p, str) for p in paths)):
        raise ValueError(f'Expected `paths` to be a `str`, `pathlib.Path`, or `list[str]`, but got `{paths}`.')
    elif len(paths) == 0:
        raise ValueError('Must provide at least one path.')
    need_unwrap_path_protocol = True
    if filesystem and (not isinstance(filesystem, FileSystem)):
        err_msg = f'The filesystem passed must either conform to pyarrow.fs.FileSystem, or fsspec.spec.AbstractFileSystem. The provided filesystem was: {filesystem}'
        try:
            import fsspec
            from fsspec.implementations.http import HTTPFileSystem
        except ModuleNotFoundError:
            raise TypeError(err_msg) from None
        if not isinstance(filesystem, fsspec.spec.AbstractFileSystem):
            raise TypeError(err_msg) from None
        if isinstance(filesystem, HTTPFileSystem):
            need_unwrap_path_protocol = False
        filesystem = PyFileSystem(FSSpecHandler(filesystem))
    resolved_paths = []
    for path in paths:
        path = _resolve_custom_scheme(path)
        try:
            resolved_filesystem, resolved_path = _resolve_filesystem_and_path(path, filesystem)
        except pa.lib.ArrowInvalid as e:
            if 'Cannot parse URI' in str(e):
                resolved_filesystem, resolved_path = _resolve_filesystem_and_path(_encode_url(path), filesystem)
                resolved_path = _decode_url(resolved_path)
            elif 'Unrecognized filesystem type in URI' in str(e):
                scheme = urllib.parse.urlparse(path, allow_fragments=False).scheme
                if scheme in ['http', 'https']:
                    try:
                        from fsspec.implementations.http import HTTPFileSystem
                    except ModuleNotFoundError:
                        raise ImportError('Please install fsspec to read files from HTTP.') from None
                    resolved_filesystem = PyFileSystem(FSSpecHandler(HTTPFileSystem()))
                    resolved_path = path
                    need_unwrap_path_protocol = False
                else:
                    raise
            else:
                raise
        if filesystem is None:
            filesystem = resolved_filesystem
        elif need_unwrap_path_protocol:
            resolved_path = _unwrap_protocol(resolved_path)
        resolved_path = filesystem.normalize_path(resolved_path)
        resolved_paths.append(resolved_path)
    return (resolved_paths, filesystem)