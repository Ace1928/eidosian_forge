from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gzip
import io
import json
import os
import string
import subprocess
import sys
import tarfile
import tempfile
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v1 import docker_creds as v1_creds
from containerregistry.client.v1 import docker_http
import httplib2
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
class FromShardedTarball(DockerImage):
    """This decodes the sharded image tarballs from docker_build."""

    def __init__(self, layer_to_tarball, top, name=None, compresslevel=9):
        self._layer_to_tarball = layer_to_tarball
        self._top = top
        self._compresslevel = compresslevel
        self._memoize = {}
        self._lock = threading.Lock()
        self._name = name

    def _content(self, layer_id, name, memoize=True):
        """Fetches a particular path's contents from the tarball."""
        if memoize:
            with self._lock:
                if name in self._memoize:
                    return self._memoize[name]
        with tarfile.open(name=self._layer_to_tarball(layer_id), mode='r:') as tar:
            try:
                content = tar.extractfile(name).read()
            except KeyError:
                content = tar.extractfile('./' + name).read()
            if memoize:
                with self._lock:
                    self._memoize[name] = content
            return content

    def top(self):
        """Override."""
        return self._top

    def repositories(self):
        """Override."""
        return json.loads(self._content(self.top(), 'repositories').decode('utf8'))

    def json(self, layer_id):
        """Override."""
        return self._content(layer_id, layer_id + '/json').decode('utf8')

    def uncompressed_layer(self, layer_id):
        """Override."""
        return self._content(layer_id, layer_id + '/layer.tar', memoize=False)

    def layer(self, layer_id):
        """Override."""
        unzipped = self.uncompressed_layer(layer_id)
        buf = io.BytesIO()
        f = gzip.GzipFile(mode='wb', compresslevel=self._compresslevel, fileobj=buf)
        try:
            f.write(unzipped)
        finally:
            f.close()
        zipped = buf.getvalue()
        return zipped

    def ancestry(self, layer_id):
        """Override."""
        p = self.parent(layer_id)
        if not p:
            return [layer_id]
        return [layer_id] + self.ancestry(p)

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        pass