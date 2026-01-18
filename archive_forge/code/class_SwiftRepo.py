import json
import os
import posixpath
import stat
import sys
import tempfile
import urllib.parse as urlparse
import zlib
from configparser import ConfigParser
from io import BytesIO
from geventhttpclient import HTTPClient
from ..greenthreads import GreenThreadsMissingObjectFinder
from ..lru_cache import LRUSizeCache
from ..object_store import INFODIR, PACKDIR, PackBasedObjectStore
from ..objects import S_ISGITLINK, Blob, Commit, Tag, Tree
from ..pack import (
from ..protocol import TCP_GIT_PORT
from ..refs import InfoRefsContainer, read_info_refs, write_info_refs
from ..repo import OBJECTDIR, BaseRepo
from ..server import Backend, TCPGitServer
class SwiftRepo(BaseRepo):

    def __init__(self, root, conf) -> None:
        """Init a Git bare Repository on top of a Swift container.

        References are managed in info/refs objects by
        `SwiftInfoRefsContainer`. The root attribute is the Swift
        container that contain the Git bare repository.

        Args:
          root: The container which contains the bare repo
          conf: A ConfigParser object
        """
        self.root = root.lstrip('/')
        self.conf = conf
        self.scon = SwiftConnector(self.root, self.conf)
        objects = self.scon.get_container_objects()
        if not objects:
            raise Exception('There is not any GIT repo here : %s' % self.root)
        objects = [o['name'].split('/')[0] for o in objects]
        if OBJECTDIR not in objects:
            raise Exception('This repository (%s) is not bare.' % self.root)
        self.bare = True
        self._controldir = self.root
        object_store = SwiftObjectStore(self.scon)
        refs = SwiftInfoRefsContainer(self.scon, object_store)
        BaseRepo.__init__(self, object_store, refs)

    def _determine_file_mode(self):
        """Probe the file-system to determine whether permissions can be trusted.

        Returns: True if permissions can be trusted, False otherwise.
        """
        return False

    def _put_named_file(self, filename, contents):
        """Put an object in a Swift container.

        Args:
          filename: the path to the object to put on Swift
          contents: the content as bytestring
        """
        with BytesIO() as f:
            f.write(contents)
            self.scon.put_object(filename, f)

    @classmethod
    def init_bare(cls, scon, conf):
        """Create a new bare repository.

        Args:
          scon: a `SwiftConnector` instance
          conf: a ConfigParser object
        Returns:
          a `SwiftRepo` instance
        """
        scon.create_root()
        for obj in [posixpath.join(OBJECTDIR, PACKDIR), posixpath.join(INFODIR, 'refs')]:
            scon.put_object(obj, BytesIO(b''))
        ret = cls(scon.root, conf)
        ret._init_files(True)
        return ret