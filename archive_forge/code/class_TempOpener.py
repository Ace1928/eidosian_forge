from __future__ import absolute_import, print_function, unicode_literals
import typing
from .base import Opener
from .registry import registry
@registry.install
class TempOpener(Opener):
    """`TempFS` opener."""
    protocols = ['temp']

    def open_fs(self, fs_url, parse_result, writeable, create, cwd):
        from ..tempfs import TempFS
        temp_fs = TempFS(identifier=parse_result.resource)
        return temp_fs