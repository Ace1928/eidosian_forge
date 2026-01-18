from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gzip
import io
import json
import os
import tarfile
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
import httplib2
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import six.moves.http_client
class FromTarball(DockerImage):
    """This decodes the image tarball output of docker_build for upload."""

    def __init__(self, tarball, name=None, compresslevel=9):
        super().__init__()
        self._tarball = tarball
        self._compresslevel = compresslevel
        self._memoize = {}
        self._lock = threading.Lock()
        self._name = name
        self._manifest = None
        self._blob_names = None
        self._config_blob = None

    def _content(self, name, memoize=True, should_be_compressed=False):
        """Fetches a particular path's contents from the tarball."""
        if memoize:
            with self._lock:
                if (name, should_be_compressed) in self._memoize:
                    return self._memoize[name, should_be_compressed]
        with tarfile.open(name=self._tarball, mode='r') as tar:
            try:
                f = tar.extractfile(str(name))
                content = f.read()
            except KeyError:
                content = tar.extractfile(str('./' + name)).read()
            if should_be_compressed and (not is_compressed(content)):
                buf = io.BytesIO()
                zipped = gzip.GzipFile(mode='wb', compresslevel=self._compresslevel, fileobj=buf)
                try:
                    zipped.write(content)
                finally:
                    zipped.close()
                content = buf.getvalue()
            elif not should_be_compressed and is_compressed(content):
                buf = io.BytesIO(content)
                raw = gzip.GzipFile(mode='rb', fileobj=buf)
                content = raw.read()
            if memoize:
                with self._lock:
                    self._memoize[name, should_be_compressed] = content
            return content

    def _gzipped_content(self, name):
        """Returns the result of _content with gzip applied."""
        return self._content(name, memoize=False, should_be_compressed=True)

    def _populate_manifest_and_blobs(self):
        """Populates self._manifest and self._blob_names."""
        config_blob = docker_digest.SHA256(self.config_file().encode('utf8'))
        manifest = {'mediaType': docker_http.MANIFEST_SCHEMA2_MIME, 'schemaVersion': 2, 'config': {'digest': config_blob, 'mediaType': docker_http.CONFIG_JSON_MIME, 'size': len(self.config_file())}, 'layers': []}
        blob_names = {}
        config = json.loads(self.config_file())
        diff_ids = config['rootfs']['diff_ids']
        for i, layer in enumerate(self._layers):
            name = None
            diff_id = diff_ids[i]
            media_type = docker_http.LAYER_MIME
            size = 0
            urls = []
            if diff_id in self._layer_sources:
                name = self._layer_sources[diff_id]['digest']
                media_type = self._layer_sources[diff_id]['mediaType']
                size = self._layer_sources[diff_id]['size']
                if 'urls' in self._layer_sources[diff_id]:
                    urls = self._layer_sources[diff_id]['urls']
            else:
                content = self._gzipped_content(layer)
                name = docker_digest.SHA256(content)
                size = len(content)
            blob_names[name] = layer
            layer_manifest = {'digest': name, 'mediaType': media_type, 'size': size}
            if urls:
                layer_manifest['urls'] = urls
            manifest['layers'].append(layer_manifest)
        with self._lock:
            self._manifest = manifest
            self._blob_names = blob_names
            self._config_blob = config_blob

    def manifest(self):
        """Override."""
        if not self._manifest:
            self._populate_manifest_and_blobs()
        return json.dumps(self._manifest, sort_keys=True)

    def config_file(self):
        """Override."""
        return self._content(self._config_file).decode('utf8')

    def uncompressed_blob(self, digest):
        """Override."""
        if not self._blob_names:
            self._populate_manifest_and_blobs()
        assert self._blob_names is not None
        return self._content(self._blob_names[digest], memoize=False, should_be_compressed=False)

    def blob(self, digest):
        """Override."""
        if not self._blob_names:
            self._populate_manifest_and_blobs()
        if digest == self._config_blob:
            return self.config_file().encode('utf8')
        assert self._blob_names is not None
        return self._gzipped_content(self._blob_names[digest])

    def uncompressed_layer(self, diff_id):
        """Override."""
        for layer, this_diff_id in zip(reversed(self._layers), self.diff_ids()):
            if diff_id == this_diff_id:
                return self._content(layer, memoize=False, should_be_compressed=False)
        raise ValueError('Unmatched "diff_id": "%s"' % diff_id)

    def _resolve_tag(self):
        """Resolve the singleton tag this tarball contains using legacy methods."""
        repo_bytes = self._content('repositories', memoize=False)
        repositories = json.loads(repo_bytes.decode('utf8'))
        if len(repositories) != 1:
            raise ValueError('Tarball must contain a single repository, or a name must be specified to FromTarball.')
        for repo, tags in six.iteritems(repositories):
            if len(tags) != 1:
                raise ValueError('Tarball must contain a single tag, or a name must be specified to FromTarball.')
            for tag, unused_layer in six.iteritems(tags):
                return '{repository}:{tag}'.format(repository=repo, tag=tag)
        raise Exception('unreachable')

    def __enter__(self):
        manifest_json = self._content('manifest.json').decode('utf8')
        manifest_list = json.loads(manifest_json)
        config = None
        layers = []
        layer_sources = []
        if len(manifest_list) != 1:
            if not self._name:
                self._name = self._resolve_tag()
        for entry in manifest_list:
            if not self._name or str(self._name) in (entry.get('RepoTags') or []):
                config = entry.get('Config')
                layers = entry.get('Layers', [])
                layer_sources = entry.get('LayerSources', {})
        if not config:
            raise ValueError('Unable to find %s in provided tarball.' % self._name)
        self._config_file = config
        self._layers = layers
        self._layer_sources = layer_sources
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        pass