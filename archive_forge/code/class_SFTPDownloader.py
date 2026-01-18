import os
import sys
import ftplib
import warnings
from .utils import parse_url
class SFTPDownloader:
    """
    Download manager for fetching files over SFTP.

    When called, downloads the given file URL into the specified local file.
    Requires `paramiko <https://github.com/paramiko/paramiko>`__ to be
    installed.

    Use with :meth:`pooch.Pooch.fetch` or :func:`pooch.retrieve` to customize
    the download of files (for example, to use authentication or print a
    progress bar).

    Parameters
    ----------
    port : int
        Port used for the SFTP connection.
    username : str
        User name used to login to the server. Only needed if the server
        requires authentication (i.e., no anonymous SFTP).
    password : str
        Password used to login to the server. Only needed if the server
        requires authentication (i.e., no anonymous SFTP). Use the empty
        string to indicate no password is required.
    timeout : int
        Timeout in seconds for sftp socket operations, use None to mean no
        timeout.
    progressbar : bool or an arbitrary progress bar object
        If True, will print a progress bar of the download to standard
        error (stderr). Requires `tqdm <https://github.com/tqdm/tqdm>`__ to
        be installed.

    """

    def __init__(self, port=22, username='anonymous', password='', account='', timeout=None, progressbar=False):
        self.port = port
        self.username = username
        self.password = password
        self.account = account
        self.timeout = timeout
        self.progressbar = progressbar
        errors = []
        if self.progressbar and tqdm is None:
            errors.append("Missing package 'tqdm' required for progress bars.")
        if paramiko is None:
            errors.append("Missing package 'paramiko' required for SFTP downloads.")
        if errors:
            raise ValueError(' '.join(errors))

    def __call__(self, url, output_file, pooch):
        """
        Download the given URL over SFTP to the given output file.

        The output file must be given as a string (file name/path) and not an
        open file object! Otherwise, paramiko cannot save to that file.

        Parameters
        ----------
        url : str
            The URL to the file you want to download.
        output_file : str
            Path (and file name) to which the file will be downloaded. **Cannot
            be a file object**.
        pooch : :class:`~pooch.Pooch`
            The instance of :class:`~pooch.Pooch` that is calling this method.
        """
        parsed_url = parse_url(url)
        connection = paramiko.Transport(sock=(parsed_url['netloc'], self.port))
        sftp = None
        try:
            connection.connect(username=self.username, password=self.password)
            sftp = paramiko.SFTPClient.from_transport(connection)
            sftp.get_channel().settimeout = self.timeout
            if self.progressbar:
                size = int(sftp.stat(parsed_url['path']).st_size)
                use_ascii = bool(sys.platform == 'win32')
                progress = tqdm(total=size, ncols=79, ascii=use_ascii, unit='B', unit_scale=True, leave=True)
            if self.progressbar:
                with progress:

                    def callback(current, total):
                        """Update the progress bar and write to output"""
                        progress.total = int(total)
                        progress.update(int(current - progress.n))
                    sftp.get(parsed_url['path'], output_file, callback=callback)
            else:
                sftp.get(parsed_url['path'], output_file)
        finally:
            connection.close()
            if sftp is not None:
                sftp.close()