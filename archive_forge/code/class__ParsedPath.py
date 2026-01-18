import pathlib
import re
import sys
from urllib.parse import urlparse
import attr
from rasterio.errors import PathError
@attr.s(slots=True)
class _ParsedPath(_Path):
    """Result of parsing a dataset URI/Path

    Attributes
    ----------
    path : str
        Parsed path. Includes the hostname and query string in the case
        of a URI.
    archive : str
        Parsed archive path.
    scheme : str
        URI scheme such as "https" or "zip+s3".
    """
    path = attr.ib()
    archive = attr.ib()
    scheme = attr.ib()

    @classmethod
    def from_uri(cls, uri):
        parts = urlparse(uri)
        path = pathlib.Path(parts.path).as_posix() if parts.path else parts.path
        scheme = parts.scheme or None
        if parts.query:
            path += '?' + parts.query
        if scheme and scheme.startswith(('gzip', 'tar', 'zip')):
            path_parts = path.split('!')
            path = path_parts.pop() if path_parts else None
            archive = path_parts.pop() if path_parts else None
        else:
            archive = None
        if parts.scheme and parts.netloc:
            if archive:
                archive = parts.netloc + archive
            else:
                path = parts.netloc + path
        return _ParsedPath(path, archive, scheme)

    @property
    def name(self):
        """The parsed path's original URI"""
        if not self.scheme:
            return self.path
        elif self.archive:
            return '{}://{}!{}'.format(self.scheme, self.archive, self.path)
        else:
            return '{}://{}'.format(self.scheme, self.path)

    @property
    def is_remote(self):
        """Test if the path is a remote, network URI"""
        return bool(self.scheme) and self.scheme.split('+')[-1] in REMOTESCHEMES

    @property
    def is_local(self):
        """Test if the path is a local URI"""
        return not self.scheme or (self.scheme and self.scheme.split('+')[-1] not in REMOTESCHEMES)