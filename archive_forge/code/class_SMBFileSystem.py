import datetime
import uuid
from stat import S_ISDIR, S_ISLNK
import smbclient
from .. import AbstractFileSystem
from ..utils import infer_storage_options
class SMBFileSystem(AbstractFileSystem):
    """Allow reading and writing to Windows and Samba network shares.

    When using `fsspec.open()` for getting a file-like object the URI
    should be specified as this format:
    ``smb://workgroup;user:password@server:port/share/folder/file.csv``.

    Example::

        >>> import fsspec
        >>> with fsspec.open(
        ...     'smb://myuser:mypassword@myserver.com/' 'share/folder/file.csv'
        ... ) as smbfile:
        ...     df = pd.read_csv(smbfile, sep='|', header=None)

    Note that you need to pass in a valid hostname or IP address for the host
    component of the URL. Do not use the Windows/NetBIOS machine name for the
    host component.

    The first component of the path in the URL points to the name of the shared
    folder. Subsequent path components will point to the directory/folder/file.

    The URL components ``workgroup`` , ``user``, ``password`` and ``port`` may be
    optional.

    .. note::

        For working this source require `smbprotocol`_ to be installed, e.g.::

            $ pip install smbprotocol
            # or
            # pip install smbprotocol[kerberos]

    .. _smbprotocol: https://github.com/jborean93/smbprotocol#requirements

    Note: if using this with the ``open`` or ``open_files``, with full URLs,
    there is no way to tell if a path is relative, so all paths are assumed
    to be absolute.
    """
    protocol = 'smb'

    def __init__(self, host, port=None, username=None, password=None, timeout=60, encrypt=None, share_access=None, **kwargs):
        """
        You can use _get_kwargs_from_urls to get some kwargs from
        a reasonable SMB url.

        Authentication will be anonymous or integrated if username/password are not
        given.

        Parameters
        ----------
        host: str
            The remote server name/ip to connect to
        port: int or None
            Port to connect with. Usually 445, sometimes 139.
        username: str or None
            Username to connect with. Required if Kerberos auth is not being used.
        password: str or None
            User's password on the server, if using username
        timeout: int
            Connection timeout in seconds
        encrypt: bool
            Whether to force encryption or not, once this has been set to True
            the session cannot be changed back to False.
        share_access: str or None
            Specifies the default access applied to file open operations
            performed with this file system object.
            This affects whether other processes can concurrently open a handle
            to the same file.

            - None (the default): exclusively locks the file until closed.
            - 'r': Allow other handles to be opened with read access.
            - 'w': Allow other handles to be opened with write access.
            - 'd': Allow other handles to be opened with delete access.
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        self.encrypt = encrypt
        self.temppath = kwargs.pop('temppath', '')
        self.share_access = share_access
        self._connect()

    @property
    def _port(self):
        return 445 if self.port is None else self.port

    def _connect(self):
        import time
        for _ in range(5):
            try:
                smbclient.register_session(self.host, username=self.username, password=self.password, port=self._port, encrypt=self.encrypt, connection_timeout=self.timeout)
                break
            except Exception:
                time.sleep(0.1)

    @classmethod
    def _strip_protocol(cls, path):
        return infer_storage_options(path)['path']

    @staticmethod
    def _get_kwargs_from_urls(path):
        out = infer_storage_options(path)
        out.pop('path', None)
        out.pop('protocol', None)
        return out

    def mkdir(self, path, create_parents=True, **kwargs):
        wpath = _as_unc_path(self.host, path)
        if create_parents:
            smbclient.makedirs(wpath, exist_ok=False, port=self._port, **kwargs)
        else:
            smbclient.mkdir(wpath, port=self._port, **kwargs)

    def makedirs(self, path, exist_ok=False):
        if _share_has_path(path):
            wpath = _as_unc_path(self.host, path)
            smbclient.makedirs(wpath, exist_ok=exist_ok, port=self._port)

    def rmdir(self, path):
        if _share_has_path(path):
            wpath = _as_unc_path(self.host, path)
            smbclient.rmdir(wpath, port=self._port)

    def info(self, path, **kwargs):
        wpath = _as_unc_path(self.host, path)
        stats = smbclient.stat(wpath, port=self._port, **kwargs)
        if S_ISDIR(stats.st_mode):
            stype = 'directory'
        elif S_ISLNK(stats.st_mode):
            stype = 'link'
        else:
            stype = 'file'
        res = {'name': path + '/' if stype == 'directory' else path, 'size': stats.st_size, 'type': stype, 'uid': stats.st_uid, 'gid': stats.st_gid, 'time': stats.st_atime, 'mtime': stats.st_mtime}
        return res

    def created(self, path):
        """Return the created timestamp of a file as a datetime.datetime"""
        wpath = _as_unc_path(self.host, path)
        stats = smbclient.stat(wpath, port=self._port)
        return datetime.datetime.fromtimestamp(stats.st_ctime, tz=datetime.timezone.utc)

    def modified(self, path):
        """Return the modified timestamp of a file as a datetime.datetime"""
        wpath = _as_unc_path(self.host, path)
        stats = smbclient.stat(wpath, port=self._port)
        return datetime.datetime.fromtimestamp(stats.st_mtime, tz=datetime.timezone.utc)

    def ls(self, path, detail=True, **kwargs):
        unc = _as_unc_path(self.host, path)
        listed = smbclient.listdir(unc, port=self._port, **kwargs)
        dirs = ['/'.join([path.rstrip('/'), p]) for p in listed]
        if detail:
            dirs = [self.info(d) for d in dirs]
        return dirs

    def _open(self, path, mode='rb', block_size=-1, autocommit=True, cache_options=None, **kwargs):
        """
        block_size: int or None
            If 0, no buffering, 1, line buffering, >1, buffer that many bytes

        Notes
        -----
        By specifying 'share_access' in 'kwargs' it is possible to override the
        default shared access setting applied in the constructor of this object.
        """
        bls = block_size if block_size is not None and block_size >= 0 else -1
        wpath = _as_unc_path(self.host, path)
        share_access = kwargs.pop('share_access', self.share_access)
        if 'w' in mode and autocommit is False:
            temp = _as_temp_path(self.host, path, self.temppath)
            return SMBFileOpener(wpath, temp, mode, port=self._port, block_size=bls, **kwargs)
        return smbclient.open_file(wpath, mode, buffering=bls, share_access=share_access, port=self._port, **kwargs)

    def copy(self, path1, path2, **kwargs):
        """Copy within two locations in the same filesystem"""
        wpath1 = _as_unc_path(self.host, path1)
        wpath2 = _as_unc_path(self.host, path2)
        smbclient.copyfile(wpath1, wpath2, port=self._port, **kwargs)

    def _rm(self, path):
        if _share_has_path(path):
            wpath = _as_unc_path(self.host, path)
            stats = smbclient.stat(wpath, port=self._port)
            if S_ISDIR(stats.st_mode):
                smbclient.rmdir(wpath, port=self._port)
            else:
                smbclient.remove(wpath, port=self._port)

    def mv(self, path1, path2, recursive=None, maxdepth=None, **kwargs):
        wpath1 = _as_unc_path(self.host, path1)
        wpath2 = _as_unc_path(self.host, path2)
        smbclient.rename(wpath1, wpath2, port=self._port, **kwargs)