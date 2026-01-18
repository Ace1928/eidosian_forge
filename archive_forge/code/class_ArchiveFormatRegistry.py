import os
import time
import warnings
from typing import Iterator, cast
from .. import errors, pyutils, registry, trace
class ArchiveFormatRegistry(registry.Registry):
    """Registry of archive formats."""

    def __init__(self):
        self._extension_map = {}
        super().__init__()

    @property
    def extensions(self):
        return self._extension_map.keys()

    def register(self, key, factory, extensions, help=None):
        """Register an archive format.
        """
        registry.Registry.register(self, key, factory, help, ArchiveFormatInfo(extensions))
        self._register_extensions(key, extensions)

    def register_lazy(self, key, module_name, member_name, extensions, help=None):
        registry.Registry.register_lazy(self, key, module_name, member_name, help, ArchiveFormatInfo(extensions))
        self._register_extensions(key, extensions)

    def _register_extensions(self, name, extensions):
        for ext in extensions:
            self._extension_map[ext] = name

    def get_format_from_filename(self, filename):
        """Determine the archive format from an extension.

        :param filename: Filename to guess from
        :return: A format name, or None
        """
        for ext, format in self._extension_map.items():
            if filename.endswith(ext):
                return format
        else:
            return None