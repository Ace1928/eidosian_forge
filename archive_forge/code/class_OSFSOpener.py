from __future__ import absolute_import, print_function, unicode_literals
import typing
from .base import Opener
from .registry import registry
@registry.install
class OSFSOpener(Opener):
    """`OSFS` opener."""
    protocols = ['file', 'osfs']

    def open_fs(self, fs_url, parse_result, writeable, create, cwd):
        from os.path import abspath, expanduser, join, normpath
        from ..osfs import OSFS
        _path = abspath(join(cwd, expanduser(parse_result.resource)))
        path = normpath(_path)
        osfs = OSFS(path, create=create)
        return osfs