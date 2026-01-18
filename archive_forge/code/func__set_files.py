import os
from paramiko.sftp import SFTP_OP_UNSUPPORTED, SFTP_OK
from paramiko.util import ClosingContextManager
from paramiko.sftp_server import SFTPServer
def _set_files(self, files):
    """
        Used by the SFTP server code to cache a directory listing.  (In
        the SFTP protocol, listing a directory is a multi-stage process
        requiring a temporary handle.)
        """
    self.__files = files