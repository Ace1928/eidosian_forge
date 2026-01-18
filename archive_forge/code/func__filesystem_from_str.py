from pyarrow.util import _is_path_like, _stringify_path
from pyarrow._fs import (  # noqa
def _filesystem_from_str(uri):
    filesystem, prefix = FileSystem.from_uri(uri)
    prefix = filesystem.normalize_path(prefix)
    if prefix:
        prefix_info = filesystem.get_file_info([prefix])[0]
        if prefix_info.type != FileType.Directory:
            raise ValueError('The path component of the filesystem URI must point to a directory but it has a type: `{}`. The path component is `{}` and the given filesystem URI is `{}`'.format(prefix_info.type.name, prefix_info.path, uri))
        filesystem = SubTreeFileSystem(prefix, filesystem)
    return filesystem