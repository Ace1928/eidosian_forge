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