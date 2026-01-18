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
def _populate_manifest(self):
    base_layers = []
    if self._legacy_base:
        base_layers = json.loads(self._legacy_base.manifest())['layers']
    elif self._foreign_layers_manifest:
        base_layers += self._get_foreign_layers()
    self._manifest = json.dumps({'schemaVersion': 2, 'mediaType': docker_http.MANIFEST_SCHEMA2_MIME, 'config': {'mediaType': docker_http.CONFIG_JSON_MIME, 'size': len(self.config_file()), 'digest': docker_digest.SHA256(self.config_file().encode('utf8'))}, 'layers': base_layers + [{'mediaType': docker_http.LAYER_MIME, 'size': self.blob_size(digest), 'digest': digest} for digest in self._layers]}, sort_keys=True)