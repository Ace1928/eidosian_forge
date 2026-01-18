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
class FromDisk(DockerImage):
    """This accesses a more efficient on-disk format than FromTarball.

  FromDisk reads an on-disk format optimized for use with push and pull.

  It is expected that the number of layers in config_file's rootfs.diff_ids
  matches: count(legacy_base.layers) + len(layers).

  Layers are drawn from legacy_base first (it is expected to be the base),
  and then from layers.

  This is effectively the dual of the save.fast method, and is intended for use
  with Bazel's rules_docker.

  Args:
    config_file: the contents of the config file.
    layers: a list of pairs.  The first element is the path to a file containing
        the second element's sha256.  The second element is the .tar.gz of a
        filesystem layer.  These are ordered as they'd appear in the manifest.
    uncompressed_layers: Optionally, a list of pairs. The first element is the
        path to a file containing the second element's sha256.
        The second element is the .tar of a filesystem layer.
    legacy_base: Optionally, the path to a legacy base image in FromTarball form
    foreign_layers_manifest: Optionally a tar manifest from the base
        image that describes the ForeignLayers needed by this image.
  """

    def __init__(self, config_file, layers, uncompressed_layers=None, legacy_base=None, foreign_layers_manifest=None):
        super().__init__()
        self._config = config_file
        self._manifest = None
        self._foreign_layers_manifest = foreign_layers_manifest
        self._layers = []
        self._layer_to_filename = {}
        for name_file, content_file in layers:
            with io.open(name_file, u'r') as reader:
                layer_name = 'sha256:' + reader.read()
            self._layers.append(layer_name)
            self._layer_to_filename[layer_name] = content_file
        self._uncompressed_layers = []
        self._uncompressed_layer_to_filename = {}
        if uncompressed_layers:
            for name_file, content_file in uncompressed_layers:
                with io.open(name_file, u'r') as reader:
                    layer_name = 'sha256:' + reader.read()
                self._uncompressed_layers.append(layer_name)
                self._uncompressed_layer_to_filename[layer_name] = content_file
        self._legacy_base = None
        if legacy_base:
            with FromTarball(legacy_base) as base:
                self._legacy_base = base

    def _get_foreign_layers(self):
        foreign_layers = []
        if self._foreign_layers_manifest:
            manifest = json.loads(self._foreign_layers_manifest)
            if 'layers' in manifest:
                for layer in manifest['layers']:
                    if layer['mediaType'] == docker_http.FOREIGN_LAYER_MIME:
                        foreign_layers.append(layer)
        return foreign_layers

    def _get_foreign_layer_by_digest(self, digest):
        for foreign_layer in self._get_foreign_layers():
            if foreign_layer['digest'] == digest:
                return foreign_layer
        return None

    def _populate_manifest(self):
        base_layers = []
        if self._legacy_base:
            base_layers = json.loads(self._legacy_base.manifest())['layers']
        elif self._foreign_layers_manifest:
            base_layers += self._get_foreign_layers()
        self._manifest = json.dumps({'schemaVersion': 2, 'mediaType': docker_http.MANIFEST_SCHEMA2_MIME, 'config': {'mediaType': docker_http.CONFIG_JSON_MIME, 'size': len(self.config_file()), 'digest': docker_digest.SHA256(self.config_file().encode('utf8'))}, 'layers': base_layers + [{'mediaType': docker_http.LAYER_MIME, 'size': self.blob_size(digest), 'digest': digest} for digest in self._layers]}, sort_keys=True)

    def manifest(self):
        """Override."""
        if not self._manifest:
            self._populate_manifest()
        assert self._manifest is not None
        return self._manifest

    def config_file(self):
        """Override."""
        return self._config

    def uncompressed_blob(self, digest):
        """Override."""
        if digest not in self._layer_to_filename:
            if self._get_foreign_layer_by_digest(digest):
                return bytes([])
            else:
                return self._checked_legacy_base.uncompressed_blob(digest)
        return super(FromDisk, self).uncompressed_blob(digest)

    def uncompressed_layer(self, diff_id):
        if diff_id in self._uncompressed_layer_to_filename:
            with io.open(self._uncompressed_layer_to_filename[diff_id], 'rb') as reader:
                return reader.read()
        if self._legacy_base and diff_id in self._legacy_base.diff_ids():
            return self._legacy_base.uncompressed_layer(diff_id)
        return super(FromDisk, self).uncompressed_layer(diff_id)

    def blob(self, digest):
        """Override."""
        if digest not in self._layer_to_filename:
            return self._checked_legacy_base.blob(digest)
        with open(self._layer_to_filename[digest], 'rb') as reader:
            return reader.read()

    def blob_size(self, digest):
        """Override."""
        if digest not in self._layer_to_filename:
            return self._checked_legacy_base.blob_size(digest)
        info = os.stat(self._layer_to_filename[digest])
        return info.st_size

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        pass

    @property
    def _checked_legacy_base(self):
        if self._legacy_base is None:
            raise ValueError('self._legacy_base is None. set legacy_base in constructor.')
        return self._legacy_base