from __future__ import absolute_import, print_function, unicode_literals
import typing
from .base import Opener
from .errors import NotWriteable
from .registry import registry
@registry.install
class ZipOpener(Opener):
    """`ZipFS` opener."""
    protocols = ['zip']

    def open_fs(self, fs_url, parse_result, writeable, create, cwd):
        from ..zipfs import ZipFS
        if not create and writeable:
            raise NotWriteable('Unable to open existing ZIP file for writing')
        zip_fs = ZipFS(parse_result.resource, write=create)
        return zip_fs