import re
import sys
import attr
from urllib.parse import urlparse
@attr.s(slots=True)
class ParsedPath(Path):
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
        path = parts.path
        scheme = parts.scheme or None
        if parts.query:
            path += '?' + parts.query
        if parts.scheme and parts.netloc:
            path = parts.netloc + path
        parts = path.split('!')
        path = parts.pop() if parts else None
        archive = parts.pop() if parts else None
        return ParsedPath(path, archive, scheme)

    @property
    def name(self):
        """The parsed path's original URI"""
        if not self.scheme:
            return self.path
        elif self.archive:
            return f'{self.scheme}://{self.archive}!{self.path}'
        else:
            return f'{self.scheme}://{self.path}'

    @property
    def is_remote(self):
        """Test if the path is a remote, network URI"""
        return self.scheme and self.scheme.split('+')[-1] in REMOTESCHEMES

    @property
    def is_local(self):
        """Test if the path is a local URI"""
        return not self.scheme or (self.scheme and self.scheme.split('+')[-1] not in REMOTESCHEMES)