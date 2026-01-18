from __future__ import (absolute_import, division, print_function)
import json
import os
import tarfile
import subprocess
import typing as t
from contextlib import contextmanager
from hashlib import sha256
from urllib.error import URLError
from urllib.parse import urldefrag
from shutil import rmtree
from tempfile import mkdtemp
from ansible.errors import AnsibleError
from ansible.galaxy import get_collections_galaxy_meta_info
from ansible.galaxy.api import should_retry_error
from ansible.galaxy.dependency_resolution.dataclasses import _GALAXY_YAML
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.api import retry_with_delays_and_condition
from ansible.module_utils.api import generate_jittered_backoff
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.yaml import yaml_load
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
import yaml
class ConcreteArtifactsManager:
    """Manager for on-disk collection artifacts.

    It is responsible for:
        * downloading remote collections from Galaxy-compatible servers and
          direct links to tarballs or SCM repositories
        * keeping track of local ones
        * keeping track of Galaxy API tokens for downloads from Galaxy'ish
          as well as the artifact hashes
        * keeping track of Galaxy API signatures for downloads from Galaxy'ish
        * caching all of above
        * retrieving the metadata out of the downloaded artifacts
    """

    def __init__(self, b_working_directory, validate_certs=True, keyring=None, timeout=60, required_signature_count=None, ignore_signature_errors=None):
        """Initialize ConcreteArtifactsManager caches and costraints."""
        self._validate_certs = validate_certs
        self._artifact_cache = {}
        self._galaxy_artifact_cache = {}
        self._artifact_meta_cache = {}
        self._galaxy_collection_cache = {}
        self._galaxy_collection_origin_cache = {}
        self._b_working_directory = b_working_directory
        self._supplemental_signature_cache = {}
        self._keyring = keyring
        self.timeout = timeout
        self._required_signature_count = required_signature_count
        self._ignore_signature_errors = ignore_signature_errors
        self._require_build_metadata = True

    @property
    def keyring(self):
        return self._keyring

    @property
    def required_successful_signature_count(self):
        return self._required_signature_count

    @property
    def ignore_signature_errors(self):
        if self._ignore_signature_errors is None:
            return []
        return self._ignore_signature_errors

    @property
    def require_build_metadata(self):
        return self._require_build_metadata

    @require_build_metadata.setter
    def require_build_metadata(self, value):
        self._require_build_metadata = value

    def get_galaxy_artifact_source_info(self, collection):
        server = collection.src.api_server
        try:
            download_url = self._galaxy_collection_cache[collection][0]
            signatures_url, signatures = self._galaxy_collection_origin_cache[collection]
        except KeyError as key_err:
            raise RuntimeError('The is no known source for {coll!s}'.format(coll=collection)) from key_err
        return {'format_version': '1.0.0', 'namespace': collection.namespace, 'name': collection.name, 'version': collection.ver, 'server': server, 'version_url': signatures_url, 'download_url': download_url, 'signatures': signatures}

    def get_galaxy_artifact_path(self, collection):
        """Given a Galaxy-stored collection, return a cached path.

        If it's not yet on disk, this method downloads the artifact first.
        """
        try:
            return self._galaxy_artifact_cache[collection]
        except KeyError:
            pass
        try:
            url, sha256_hash, token = self._galaxy_collection_cache[collection]
        except KeyError as key_err:
            raise RuntimeError('There is no known source for {coll!s}'.format(coll=collection)) from key_err
        display.vvvv("Fetching a collection tarball for '{collection!s}' from Ansible Galaxy".format(collection=collection))
        try:
            b_artifact_path = _download_file(url, self._b_working_directory, expected_hash=sha256_hash, validate_certs=self._validate_certs, token=token)
        except URLError as err:
            raise AnsibleError("Failed to download collection tar from '{coll_src!s}': {download_err!s}".format(coll_src=to_native(collection.src), download_err=to_native(err))) from err
        except Exception as err:
            raise AnsibleError("Failed to download collection tar from '{coll_src!s}' due to the following unforeseen error: {download_err!s}".format(coll_src=to_native(collection.src), download_err=to_native(err))) from err
        else:
            display.vvv("Collection '{coll!s}' obtained from server {server!s} {url!s}".format(coll=collection, server=collection.src or 'Galaxy', url=collection.src.api_server if collection.src is not None else ''))
        self._galaxy_artifact_cache[collection] = b_artifact_path
        return b_artifact_path

    def get_artifact_path(self, collection):
        """Given a concrete collection pointer, return a cached path.

        If it's not yet on disk, this method downloads the artifact first.
        """
        try:
            return self._artifact_cache[collection.src]
        except KeyError:
            pass
        if collection.is_url:
            display.vvvv("Collection requirement '{collection!s}' is a URL to a tar artifact".format(collection=collection.fqcn))
            try:
                b_artifact_path = _download_file(collection.src, self._b_working_directory, expected_hash=None, validate_certs=self._validate_certs, timeout=self.timeout)
            except Exception as err:
                raise AnsibleError("Failed to download collection tar from '{coll_src!s}': {download_err!s}".format(coll_src=to_native(collection.src), download_err=to_native(err))) from err
        elif collection.is_scm:
            b_artifact_path = _extract_collection_from_git(collection.src, collection.ver, self._b_working_directory)
        elif collection.is_file or collection.is_dir or collection.is_subdirs:
            b_artifact_path = to_bytes(collection.src)
        else:
            raise RuntimeError('The artifact is of an unexpected type {art_type!s}'.format(art_type=collection.type))
        self._artifact_cache[collection.src] = b_artifact_path
        return b_artifact_path

    def get_artifact_path_from_unknown(self, collection):
        if collection.is_concrete_artifact:
            return self.get_artifact_path(collection)
        return self.get_galaxy_artifact_path(collection)

    def _get_direct_collection_namespace(self, collection):
        return self.get_direct_collection_meta(collection)['namespace']

    def _get_direct_collection_name(self, collection):
        return self.get_direct_collection_meta(collection)['name']

    def get_direct_collection_fqcn(self, collection):
        """Extract FQCN from the given on-disk collection artifact.

        If the collection is virtual, ``None`` is returned instead
        of a string.
        """
        if collection.is_virtual:
            return None
        return '.'.join((self._get_direct_collection_namespace(collection), self._get_direct_collection_name(collection)))

    def get_direct_collection_version(self, collection):
        """Extract version from the given on-disk collection artifact."""
        return self.get_direct_collection_meta(collection)['version']

    def get_direct_collection_dependencies(self, collection):
        """Extract deps from the given on-disk collection artifact."""
        collection_dependencies = self.get_direct_collection_meta(collection)['dependencies']
        if collection_dependencies is None:
            collection_dependencies = {}
        return collection_dependencies

    def get_direct_collection_meta(self, collection):
        """Extract meta from the given on-disk collection artifact."""
        try:
            return self._artifact_meta_cache[collection.src]
        except KeyError:
            b_artifact_path = self.get_artifact_path(collection)
        if collection.is_url or collection.is_file:
            collection_meta = _get_meta_from_tar(b_artifact_path)
        elif collection.is_dir:
            try:
                collection_meta = _get_meta_from_dir(b_artifact_path, self.require_build_metadata)
            except LookupError as lookup_err:
                raise AnsibleError('Failed to find the collection dir deps: {err!s}'.format(err=to_native(lookup_err))) from lookup_err
        elif collection.is_scm:
            collection_meta = {'name': None, 'namespace': None, 'dependencies': {to_native(b_artifact_path): '*'}, 'version': '*'}
        elif collection.is_subdirs:
            collection_meta = {'name': None, 'namespace': None, 'dependencies': dict.fromkeys(map(to_native, collection.namespace_collection_paths), '*'), 'version': '*'}
        else:
            raise RuntimeError
        self._artifact_meta_cache[collection.src] = collection_meta
        return collection_meta

    def save_collection_source(self, collection, url, sha256_hash, token, signatures_url, signatures):
        """Store collection URL, SHA256 hash and Galaxy API token.

        This is a hook that is supposed to be called before attempting to
        download Galaxy-based collections with ``get_galaxy_artifact_path()``.
        """
        self._galaxy_collection_cache[collection] = (url, sha256_hash, token)
        self._galaxy_collection_origin_cache[collection] = (signatures_url, signatures)

    @classmethod
    @contextmanager
    def under_tmpdir(cls, temp_dir_base, validate_certs=True, keyring=None, required_signature_count=None, ignore_signature_errors=None, require_build_metadata=True):
        """Custom ConcreteArtifactsManager constructor with temp dir.

        This method returns a context manager that allocates and cleans
        up a temporary directory for caching the collection artifacts
        during the dependency resolution process.
        """
        temp_path = mkdtemp(dir=to_bytes(temp_dir_base, errors='surrogate_or_strict'))
        b_temp_path = to_bytes(temp_path, errors='surrogate_or_strict')
        try:
            yield cls(b_temp_path, validate_certs, keyring=keyring, required_signature_count=required_signature_count, ignore_signature_errors=ignore_signature_errors)
        finally:
            rmtree(b_temp_path)