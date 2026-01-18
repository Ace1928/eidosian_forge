import functools
import itertools
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import warnings
import zipfile
from hashlib import md5
from xml.etree import ElementTree
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import nltk
def _update_index(self, url=None):
    """A helper function that ensures that self._index is
        up-to-date.  If the index is older than self.INDEX_TIMEOUT,
        then download it again."""
    if not (self._index is None or url is not None or time.time() - self._index_timestamp > self.INDEX_TIMEOUT):
        return
    self._url = url or self._url
    self._index = nltk.internals.ElementWrapper(ElementTree.parse(urlopen(self._url)).getroot())
    self._index_timestamp = time.time()
    packages = [Package.fromxml(p) for p in self._index.findall('packages/package')]
    self._packages = {p.id: p for p in packages}
    collections = [Collection.fromxml(c) for c in self._index.findall('collections/collection')]
    self._collections = {c.id: c for c in collections}
    for collection in self._collections.values():
        for i, child_id in enumerate(collection.children):
            if child_id in self._packages:
                collection.children[i] = self._packages[child_id]
            elif child_id in self._collections:
                collection.children[i] = self._collections[child_id]
            else:
                print('removing collection member with no package: {}'.format(child_id))
                del collection.children[i]
    for collection in self._collections.values():
        packages = {}
        queue = [collection]
        for child in queue:
            if isinstance(child, Collection):
                queue.extend(child.children)
            elif isinstance(child, Package):
                packages[child.id] = child
            else:
                pass
        collection.packages = packages.values()
    self._status_cache.clear()