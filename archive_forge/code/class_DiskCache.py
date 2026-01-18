from __future__ import annotations
import errno
import hashlib
import json
import logging
import os
import sys
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import (
import requests
from filelock import FileLock
class DiskCache:
    """Disk _cache that only works for jsonable values."""

    def __init__(self, cache_dir: str | None, lock_timeout: int=20):
        """Construct a disk cache in the given directory."""
        self.enabled = bool(cache_dir)
        self.cache_dir = os.path.expanduser(str(cache_dir) or '')
        self.lock_timeout = lock_timeout
        self.file_ext = '.tldextract.json'

    def get(self, namespace: str, key: str | dict[str, Hashable]) -> object:
        """Retrieve a value from the disk cache."""
        if not self.enabled:
            raise KeyError('Cache is disabled')
        cache_filepath = self._key_to_cachefile_path(namespace, key)
        if not os.path.isfile(cache_filepath):
            raise KeyError('namespace: ' + namespace + ' key: ' + repr(key))
        try:
            with open(cache_filepath) as cache_file:
                return json.load(cache_file)
        except (OSError, ValueError) as exc:
            LOG.error('error reading TLD cache file %s: %s', cache_filepath, exc)
            raise KeyError('namespace: ' + namespace + ' key: ' + repr(key)) from None

    def set(self, namespace: str, key: str | dict[str, Hashable], value: object) -> None:
        """Set a value in the disk cache."""
        if not self.enabled:
            return
        cache_filepath = self._key_to_cachefile_path(namespace, key)
        try:
            _make_dir(cache_filepath)
            with open(cache_filepath, 'w') as cache_file:
                json.dump(value, cache_file)
        except OSError as ioe:
            global _DID_LOG_UNABLE_TO_CACHE
            if not _DID_LOG_UNABLE_TO_CACHE:
                LOG.warning('unable to cache %s.%s in %s. This could refresh the Public Suffix List over HTTP every app startup. Construct your `TLDExtract` with a writable `cache_dir` or set `cache_dir=None` to silence this warning. %s', namespace, key, cache_filepath, ioe)
                _DID_LOG_UNABLE_TO_CACHE = True

    def clear(self) -> None:
        """Clear the disk cache."""
        for root, _, files in os.walk(self.cache_dir):
            for filename in files:
                if filename.endswith(self.file_ext) or filename.endswith(self.file_ext + '.lock'):
                    try:
                        os.unlink(str(Path(root, filename)))
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        if exc.errno != errno.ENOENT:
                            raise

    def _key_to_cachefile_path(self, namespace: str, key: str | dict[str, Hashable]) -> str:
        namespace_path = str(Path(self.cache_dir, namespace))
        hashed_key = _make_cache_key(key)
        cache_path = str(Path(namespace_path, hashed_key + self.file_ext))
        return cache_path

    def run_and_cache(self, func: Callable[..., T], namespace: str, kwargs: dict[str, Hashable], hashed_argnames: Iterable[str]) -> T:
        """Get a url but cache the response."""
        if not self.enabled:
            return func(**kwargs)
        key_args = {k: v for k, v in kwargs.items() if k in hashed_argnames}
        cache_filepath = self._key_to_cachefile_path(namespace, key_args)
        lock_path = cache_filepath + '.lock'
        try:
            _make_dir(cache_filepath)
        except OSError as ioe:
            global _DID_LOG_UNABLE_TO_CACHE
            if not _DID_LOG_UNABLE_TO_CACHE:
                LOG.warning('unable to cache %s.%s in %s. This could refresh the Public Suffix List over HTTP every app startup. Construct your `TLDExtract` with a writable `cache_dir` or set `cache_dir=None` to silence this warning. %s', namespace, key_args, cache_filepath, ioe)
                _DID_LOG_UNABLE_TO_CACHE = True
            return func(**kwargs)
        with FileLock(lock_path, timeout=self.lock_timeout):
            try:
                result = cast(T, self.get(namespace=namespace, key=key_args))
            except KeyError:
                result = func(**kwargs)
                self.set(namespace=namespace, key=key_args, value=result)
            return result

    def cached_fetch_url(self, session: requests.Session, url: str, timeout: float | int | None) -> str:
        """Get a url but cache the response."""
        return self.run_and_cache(func=_fetch_url, namespace='urls', kwargs={'session': session, 'url': url, 'timeout': timeout}, hashed_argnames=['url'])