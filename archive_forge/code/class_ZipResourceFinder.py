from __future__ import unicode_literals
import bisect
import io
import logging
import os
import pkgutil
import sys
import types
import zipimport
from . import DistlibException
from .util import cached_property, get_cache_base, Cache
class ZipResourceFinder(ResourceFinder):
    """
    Resource finder for resources in .zip files.
    """

    def __init__(self, module):
        super(ZipResourceFinder, self).__init__(module)
        archive = self.loader.archive
        self.prefix_len = 1 + len(archive)
        if hasattr(self.loader, '_files'):
            self._files = self.loader._files
        else:
            self._files = zipimport._zip_directory_cache[archive]
        self.index = sorted(self._files)

    def _adjust_path(self, path):
        return path

    def _find(self, path):
        path = path[self.prefix_len:]
        if path in self._files:
            result = True
        else:
            if path and path[-1] != os.sep:
                path = path + os.sep
            i = bisect.bisect(self.index, path)
            try:
                result = self.index[i].startswith(path)
            except IndexError:
                result = False
        if not result:
            logger.debug('_find failed: %r %r', path, self.loader.prefix)
        else:
            logger.debug('_find worked: %r %r', path, self.loader.prefix)
        return result

    def get_cache_info(self, resource):
        prefix = self.loader.archive
        path = resource.path[1 + len(prefix):]
        return (prefix, path)

    def get_bytes(self, resource):
        return self.loader.get_data(resource.path)

    def get_stream(self, resource):
        return io.BytesIO(self.get_bytes(resource))

    def get_size(self, resource):
        path = resource.path[self.prefix_len:]
        return self._files[path][3]

    def get_resources(self, resource):
        path = resource.path[self.prefix_len:]
        if path and path[-1] != os.sep:
            path += os.sep
        plen = len(path)
        result = set()
        i = bisect.bisect(self.index, path)
        while i < len(self.index):
            if not self.index[i].startswith(path):
                break
            s = self.index[i][plen:]
            result.add(s.split(os.sep, 1)[0])
            i += 1
        return result

    def _is_directory(self, path):
        path = path[self.prefix_len:]
        if path and path[-1] != os.sep:
            path += os.sep
        i = bisect.bisect(self.index, path)
        try:
            result = self.index[i].startswith(path)
        except IndexError:
            result = False
        return result