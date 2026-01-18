import abc
import os
import bz2
import gzip
import lzma
import shutil
from zipfile import ZipFile
from tarfile import TarFile
from .utils import get_logger
class ExtractorProcessor(abc.ABC):
    """
    Abstract base class for extractions from compressed archives.

    Subclasses can be used with :meth:`pooch.Pooch.fetch` and
    :func:`pooch.retrieve` to unzip a downloaded data file into a folder in the
    local data store. :meth:`~pooch.Pooch.fetch` will return a list with the
    names of the extracted files instead of the archive.

    Parameters
    ----------
    members : list or None
        If None, will unpack all files in the archive. Otherwise, *members*
        must be a list of file names to unpack from the archive. Only these
        files will be unpacked.
    extract_dir : str or None
        If None, files will be unpacked to the default location (a folder in
        the same location as the downloaded zip file, with a suffix added).
        Otherwise, files will be unpacked to ``extract_dir``, which is
        interpreted as a *relative path* (relative to the cache location
        provided by :func:`pooch.retrieve` or :meth:`pooch.Pooch.fetch`).

    """

    def __init__(self, members=None, extract_dir=None):
        self.members = members
        self.extract_dir = extract_dir

    @property
    @abc.abstractmethod
    def suffix(self):
        """
        String appended to unpacked archive folder name.
        Only used if extract_dir is None.
        MUST BE IMPLEMENTED BY CHILD CLASSES.
        """

    @abc.abstractmethod
    def _all_members(self, fname):
        """
        Return all the members in the archive.
        MUST BE IMPLEMENTED BY CHILD CLASSES.
        """

    @abc.abstractmethod
    def _extract_file(self, fname, extract_dir):
        """
        This method receives an argument for the archive to extract and the
        destination path.
        MUST BE IMPLEMENTED BY CHILD CLASSES.
        """

    def __call__(self, fname, action, pooch):
        """
        Extract all files from the given archive.

        Parameters
        ----------
        fname : str
            Full path of the zipped file in local storage.
        action : str
            Indicates what action was taken by :meth:`pooch.Pooch.fetch` or
            :func:`pooch.retrieve`:

            * ``"download"``: File didn't exist locally and was downloaded
            * ``"update"``: Local file was outdated and was re-download
            * ``"fetch"``: File exists and is updated so it wasn't downloaded

        pooch : :class:`pooch.Pooch`
            The instance of :class:`pooch.Pooch` that is calling this.

        Returns
        -------
        fnames : list of str
            A list of the full path to all files in the extracted archive.

        """
        if self.extract_dir is None:
            self.extract_dir = fname + self.suffix
        else:
            archive_dir = fname.rsplit(os.path.sep, maxsplit=1)[0]
            self.extract_dir = os.path.join(archive_dir, self.extract_dir)
        if self.members is None or not self.members:
            members = self._all_members(fname)
        else:
            members = self.members
        if action in ('update', 'download') or not os.path.exists(self.extract_dir) or (not all((os.path.exists(os.path.join(self.extract_dir, m)) for m in members))):
            os.makedirs(self.extract_dir, exist_ok=True)
            self._extract_file(fname, self.extract_dir)
        fnames = []
        for path, _, files in os.walk(self.extract_dir):
            for filename in files:
                relpath = os.path.normpath(os.path.join(os.path.relpath(path, self.extract_dir), filename))
                if self.members is None or any((relpath.startswith(os.path.normpath(m)) for m in self.members)):
                    fnames.append(os.path.join(path, filename))
        return fnames