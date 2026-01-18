from __future__ import absolute_import, print_function, unicode_literals
import typing
from .base import Opener
from .registry import registry
@registry.install
class MemOpener(Opener):
    """`MemoryFS` opener."""
    protocols = ['mem']

    def open_fs(self, fs_url, parse_result, writeable, create, cwd):
        from ..memoryfs import MemoryFS
        mem_fs = MemoryFS()
        return mem_fs