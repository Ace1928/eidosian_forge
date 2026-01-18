import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like
from pyarrow.compute import Expression, scalar, field  # noqa
def _ensure_single_source(path, filesystem=None):
    """
    Treat path as either a recursively traversable directory or a single file.

    Parameters
    ----------
    path : path-like
    filesystem : FileSystem or str, optional
        If an URI is passed, then its path component will act as a prefix for
        the file paths.

    Returns
    -------
    (FileSystem, list of str or fs.Selector)
        File system object and either a single item list pointing to a file or
        an fs.Selector object pointing to a directory.

    Raises
    ------
    TypeError
        If the passed filesystem has wrong type.
    FileNotFoundError
        If the referenced file or directory doesn't exist.
    """
    from pyarrow.fs import FileType, FileSelector, _resolve_filesystem_and_path
    filesystem, path = _resolve_filesystem_and_path(path, filesystem)
    path = filesystem.normalize_path(path)
    file_info = filesystem.get_file_info(path)
    if file_info.type == FileType.Directory:
        paths_or_selector = FileSelector(path, recursive=True)
    elif file_info.type == FileType.File:
        paths_or_selector = [path]
    else:
        raise FileNotFoundError(path)
    return (filesystem, paths_or_selector)