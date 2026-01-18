import os
import time
import contextlib
from pathlib import Path
import shlex
import shutil
from .hashes import hash_matches, file_hash
from .utils import (
from .downloaders import DOIDownloader, choose_downloader, doi_to_repository
class Pooch:
    """
    Manager for a local data storage that can fetch from a remote source.

    Avoid creating ``Pooch`` instances directly. Use :func:`pooch.create`
    instead.

    Parameters
    ----------
    path : str
        The path to the local data storage folder. The path must exist in the
        file system.
    base_url : str
        Base URL for the remote data source. All requests will be made relative
        to this URL.
    registry : dict or None
        A record of the files that are managed by this good boy. Keys should be
        the file names and the values should be their hashes. Only files
        in the registry can be fetched from the local storage. Files in
        subdirectories of *path* **must use Unix-style separators** (``'/'``)
        even on Windows.
    urls : dict or None
        Custom URLs for downloading individual files in the registry. A
        dictionary with the file names as keys and the custom URLs as values.
        Not all files in *registry* need an entry in *urls*. If a file has an
        entry in *urls*, the *base_url* will be ignored when downloading it in
        favor of ``urls[fname]``.
    retry_if_failed : int
        Retry a file download the specified number of times if it fails because
        of a bad connection or a hash mismatch. By default, downloads are only
        attempted once (``retry_if_failed=0``). Initially, will wait for 1s
        between retries and then increase the wait time by 1s with each retry
        until a maximum of 10s.
    allow_updates : bool
        Whether existing files in local storage that have a hash mismatch with
        the registry are allowed to update from the remote URL. If ``False``,
        any mismatch with hashes in the registry will result in an error.
        Defaults to ``True``.

    """

    def __init__(self, path, base_url, registry=None, urls=None, retry_if_failed=0, allow_updates=True):
        self.path = path
        self.base_url = base_url
        if registry is None:
            registry = {}
        self.registry = registry
        if urls is None:
            urls = {}
        self.urls = dict(urls)
        self.retry_if_failed = retry_if_failed
        self.allow_updates = allow_updates

    @property
    def abspath(self):
        """Absolute path to the local storage"""
        return Path(os.path.abspath(os.path.expanduser(str(self.path))))

    @property
    def registry_files(self):
        """List of file names on the registry"""
        return list(self.registry)

    def fetch(self, fname, processor=None, downloader=None, progressbar=False):
        """
        Get the absolute path to a file in the local storage.

        If it's not in the local storage, it will be downloaded. If the hash of
        the file in local storage doesn't match the one in the registry, will
        download a new copy of the file. This is considered a sign that the
        file was updated in the remote storage. If the hash of the downloaded
        file still doesn't match the one in the registry, will raise an
        exception to warn of possible file corruption.

        Post-processing actions sometimes need to be taken on downloaded files
        (unzipping, conversion to a more efficient format, etc). If these
        actions are time or memory consuming, it would be best to do this only
        once right after the file is downloaded. Use the *processor* argument
        to specify a function that is executed after the download to perform
        these actions. See :ref:`processors` for details.

        Custom file downloaders can be provided through the *downloader*
        argument. By default, Pooch will determine the download protocol from
        the URL in the registry. If the server for a given file requires
        authentication (username and password), use a downloader that support
        these features. Downloaders can also be used to print custom messages
        (like a progress bar), etc. See :ref:`downloaders` for details.

        Parameters
        ----------
        fname : str
            The file name (relative to the *base_url* of the remote data
            storage) to fetch from the local storage.
        processor : None or callable
            If not None, then a function (or callable object) that will be
            called before returning the full path and after the file has been
            downloaded. See :ref:`processors` for details.
        downloader : None or callable
            If not None, then a function (or callable object) that will be
            called to download a given URL to a provided local file name. See
            :ref:`downloaders` for details.
        progressbar : bool or an arbitrary progress bar object
            If True, will print a progress bar of the download to standard
            error (stderr). Requires `tqdm <https://github.com/tqdm/tqdm>`__ to
            be installed. Alternatively, an arbitrary progress bar object can
            be passed. See :ref:`custom-progressbar` for details.

        Returns
        -------
        full_path : str
            The absolute path (including the file name) of the file in the
            local storage.

        """
        self._assert_file_in_registry(fname)
        url = self.get_url(fname)
        full_path = self.abspath / fname
        known_hash = self.registry[fname]
        action, verb = download_action(full_path, known_hash)
        if action == 'update' and (not self.allow_updates):
            raise ValueError(f'{fname} needs to update {full_path} but updates are disallowed.')
        if action in ('download', 'update'):
            make_local_storage(str(self.abspath))
            get_logger().info("%s file '%s' from '%s' to '%s'.", verb, fname, url, str(self.abspath))
            if downloader is None:
                downloader = choose_downloader(url, progressbar=progressbar)
            stream_download(url, full_path, known_hash, downloader, pooch=self, retry_if_failed=self.retry_if_failed)
        if processor is not None:
            return processor(str(full_path), action, self)
        return str(full_path)

    def _assert_file_in_registry(self, fname):
        """
        Check if a file is in the registry and raise :class:`ValueError` if
        it's not.
        """
        if fname not in self.registry:
            raise ValueError(f"File '{fname}' is not in the registry.")

    def get_url(self, fname):
        """
        Get the full URL to download a file in the registry.

        Parameters
        ----------
        fname : str
            The file name (relative to the *base_url* of the remote data
            storage) to fetch from the local storage.

        """
        self._assert_file_in_registry(fname)
        return self.urls.get(fname, ''.join([self.base_url, fname]))

    def load_registry(self, fname):
        """
        Load entries from a file and add them to the registry.

        Use this if you are managing many files.

        Each line of the file should have file name and its hash separated by
        a space. Hash can specify checksum algorithm using "alg:hash" format.
        In case no algorithm is provided, SHA256 is used by default.
        Only one file per line is allowed. Custom download URLs for individual
        files can be specified as a third element on the line. Line comments
        can be added and must be prepended with ``#``.

        Parameters
        ----------
        fname : str | fileobj
            Path (or open file object) to the registry file.

        """
        with contextlib.ExitStack() as stack:
            if hasattr(fname, 'read'):
                fin = fname
            else:
                fin = stack.enter_context(open(fname, encoding='utf-8'))
            for linenum, line in enumerate(fin):
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                line = line.strip()
                if line.startswith('#'):
                    continue
                elements = shlex.split(line)
                if not len(elements) in [0, 2, 3]:
                    raise OSError(f"Invalid entry in Pooch registry file '{fname}': expected 2 or 3 elements in line {linenum + 1} but got {len(elements)}. Offending entry: '{line}'")
                if elements:
                    file_name = elements[0]
                    file_checksum = elements[1]
                    if len(elements) == 3:
                        file_url = elements[2]
                        self.urls[file_name] = file_url
                    self.registry[file_name] = file_checksum.lower()

    def load_registry_from_doi(self):
        """
        Populate the registry using the data repository API

        Fill the registry with all the files available in the data repository,
        along with their hashes. It will make a request to the data repository
        API to retrieve this information. No file is downloaded during this
        process.

        .. important::

            This method is intended to be used only when the ``base_url`` is
            a DOI.
        """
        downloader = choose_downloader(self.base_url)
        if not isinstance(downloader, DOIDownloader):
            raise ValueError(f"Invalid base_url '{self.base_url}': " + 'Pooch.load_registry_from_doi is only implemented for DOIs')
        doi = self.base_url.replace('doi:', '')
        repository = doi_to_repository(doi)
        return repository.populate_registry(self)

    def is_available(self, fname, downloader=None):
        """
        Check availability of a remote file without downloading it.

        Use this method when working with large files to check if they are
        available for download.

        Parameters
        ----------
        fname : str
            The file name (relative to the *base_url* of the remote data
            storage).
        downloader : None or callable
            If not None, then a function (or callable object) that will be
            called to check the availability of the file on the server. See
            :ref:`downloaders` for details.

        Returns
        -------
        status : bool
            True if the file is available for download. False otherwise.

        """
        self._assert_file_in_registry(fname)
        url = self.get_url(fname)
        if downloader is None:
            downloader = choose_downloader(url)
        try:
            available = downloader(url, None, self, check_only=True)
        except TypeError as error:
            error_msg = f"Downloader '{str(downloader)}' does not support availability checks."
            raise NotImplementedError(error_msg) from error
        return available