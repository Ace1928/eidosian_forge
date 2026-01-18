import os
import sys
import ftplib
import warnings
from .utils import parse_url
class FTPDownloader:
    """
    Download manager for fetching files over FTP.

    When called, downloads the given file URL into the specified local file.
    Uses the :mod:`ftplib` module to manage downloads.

    Use with :meth:`pooch.Pooch.fetch` or :func:`pooch.retrieve` to customize
    the download of files (for example, to use authentication or print a
    progress bar).

    Parameters
    ----------
    port : int
        Port used for the FTP connection.
    username : str
        User name used to login to the server. Only needed if the server
        requires authentication (i.e., no anonymous FTP).
    password : str
        Password used to login to the server. Only needed if the server
        requires authentication (i.e., no anonymous FTP). Use the empty string
        to indicate no password is required.
    account : str
        Some servers also require an "account" name for authentication.
    timeout : int
        Timeout in seconds for ftp socket operations, use None to mean no
        timeout.
    progressbar : bool
        If True, will print a progress bar of the download to standard error
        (stderr). Requires `tqdm <https://github.com/tqdm/tqdm>`__ to be
        installed. **Custom progress bars are not yet supported.**
    chunk_size : int
        Files are streamed *chunk_size* bytes at a time instead of loading
        everything into memory at one. Usually doesn't need to be changed.

    """

    def __init__(self, port=21, username='anonymous', password='', account='', timeout=None, progressbar=False, chunk_size=1024):
        self.port = port
        self.username = username
        self.password = password
        self.account = account
        self.timeout = timeout
        self.progressbar = progressbar
        self.chunk_size = chunk_size
        if self.progressbar is True and tqdm is None:
            raise ValueError("Missing package 'tqdm' required for progress bars.")

    def __call__(self, url, output_file, pooch, check_only=False):
        """
        Download the given URL over FTP to the given output file.

        Parameters
        ----------
        url : str
            The URL to the file you want to download.
        output_file : str or file-like object
            Path (and file name) to which the file will be downloaded.
        pooch : :class:`~pooch.Pooch`
            The instance of :class:`~pooch.Pooch` that is calling this method.
        check_only : bool
            If True, will only check if a file exists on the server and
            **without downloading the file**. Will return ``True`` if the file
            exists and ``False`` otherwise.

        Returns
        -------
        availability : bool or None
            If ``check_only==True``, returns a boolean indicating if the file
            is available on the server. Otherwise, returns ``None``.

        """
        parsed_url = parse_url(url)
        ftp = ftplib.FTP(timeout=self.timeout)
        ftp.connect(host=parsed_url['netloc'], port=self.port)
        if check_only:
            directory, file_name = os.path.split(parsed_url['path'])
            try:
                ftp.login(user=self.username, passwd=self.password, acct=self.account)
                available = file_name in ftp.nlst(directory)
            finally:
                ftp.close()
            return available
        ispath = not hasattr(output_file, 'write')
        if ispath:
            output_file = open(output_file, 'w+b')
        try:
            ftp.login(user=self.username, passwd=self.password, acct=self.account)
            command = f'RETR {parsed_url['path']}'
            if self.progressbar:
                ftp.voidcmd('TYPE I')
                use_ascii = bool(sys.platform == 'win32')
                progress = tqdm(total=int(ftp.size(parsed_url['path'])), ncols=79, ascii=use_ascii, unit='B', unit_scale=True, leave=True)
                with progress:

                    def callback(data):
                        """Update the progress bar and write to output"""
                        progress.update(len(data))
                        output_file.write(data)
                    ftp.retrbinary(command, callback, blocksize=self.chunk_size)
            else:
                ftp.retrbinary(command, output_file.write, blocksize=self.chunk_size)
        finally:
            ftp.quit()
            if ispath:
                output_file.close()
        return None