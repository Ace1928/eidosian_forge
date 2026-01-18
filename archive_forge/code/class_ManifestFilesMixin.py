import json
import os
import posixpath
import re
from hashlib import md5
from urllib.parse import unquote, urldefrag, urlsplit, urlunsplit
from django.conf import STATICFILES_STORAGE_ALIAS, settings
from django.contrib.staticfiles.utils import check_settings, matches_patterns
from django.core.exceptions import ImproperlyConfigured
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, storages
from django.utils.functional import LazyObject
class ManifestFilesMixin(HashedFilesMixin):
    manifest_version = '1.1'
    manifest_name = 'staticfiles.json'
    manifest_strict = True
    keep_intermediate_files = False

    def __init__(self, *args, manifest_storage=None, **kwargs):
        super().__init__(*args, **kwargs)
        if manifest_storage is None:
            manifest_storage = self
        self.manifest_storage = manifest_storage
        self.hashed_files, self.manifest_hash = self.load_manifest()

    def read_manifest(self):
        try:
            with self.manifest_storage.open(self.manifest_name) as manifest:
                return manifest.read().decode()
        except FileNotFoundError:
            return None

    def load_manifest(self):
        content = self.read_manifest()
        if content is None:
            return ({}, '')
        try:
            stored = json.loads(content)
        except json.JSONDecodeError:
            pass
        else:
            version = stored.get('version')
            if version in ('1.0', '1.1'):
                return (stored.get('paths', {}), stored.get('hash', ''))
        raise ValueError("Couldn't load manifest '%s' (version %s)" % (self.manifest_name, self.manifest_version))

    def post_process(self, *args, **kwargs):
        self.hashed_files = {}
        yield from super().post_process(*args, **kwargs)
        if not kwargs.get('dry_run'):
            self.save_manifest()

    def save_manifest(self):
        self.manifest_hash = self.file_hash(None, ContentFile(json.dumps(sorted(self.hashed_files.items())).encode()))
        payload = {'paths': self.hashed_files, 'version': self.manifest_version, 'hash': self.manifest_hash}
        if self.manifest_storage.exists(self.manifest_name):
            self.manifest_storage.delete(self.manifest_name)
        contents = json.dumps(payload).encode()
        self.manifest_storage._save(self.manifest_name, ContentFile(contents))

    def stored_name(self, name):
        parsed_name = urlsplit(unquote(name))
        clean_name = parsed_name.path.strip()
        hash_key = self.hash_key(clean_name)
        cache_name = self.hashed_files.get(hash_key)
        if cache_name is None:
            if self.manifest_strict:
                raise ValueError("Missing staticfiles manifest entry for '%s'" % clean_name)
            cache_name = self.clean_name(self.hashed_name(name))
        unparsed_name = list(parsed_name)
        unparsed_name[2] = cache_name
        if '?#' in name and (not unparsed_name[3]):
            unparsed_name[2] += '?'
        return urlunsplit(unparsed_name)