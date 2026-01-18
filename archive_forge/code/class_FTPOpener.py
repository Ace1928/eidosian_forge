from __future__ import absolute_import, print_function, unicode_literals
import typing
from ..errors import CreateFailed
from .base import Opener
from .registry import registry
@registry.install
class FTPOpener(Opener):
    """`FTPFS` opener."""
    protocols = ['ftp', 'ftps']

    @CreateFailed.catch_all
    def open_fs(self, fs_url, parse_result, writeable, create, cwd):
        from ..ftpfs import FTPFS
        from ..subfs import ClosingSubFS
        ftp_host, _, dir_path = parse_result.resource.partition('/')
        ftp_host, _, ftp_port = ftp_host.partition(':')
        ftp_port = int(ftp_port) if ftp_port.isdigit() else 21
        ftp_fs = FTPFS(ftp_host, port=ftp_port, user=parse_result.username, passwd=parse_result.password, proxy=parse_result.params.get('proxy'), timeout=int(parse_result.params.get('timeout', '10')), tls=bool(parse_result.protocol == 'ftps'))
        if dir_path:
            if create:
                ftp_fs.makedirs(dir_path, recreate=True)
            return ftp_fs.opendir(dir_path, factory=ClosingSubFS)
        else:
            return ftp_fs