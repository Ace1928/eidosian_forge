from __future__ import print_function, unicode_literals
import typing
import array
import calendar
import datetime
import io
import itertools
import socket
import threading
from collections import OrderedDict
from contextlib import contextmanager
from ftplib import FTP
from typing import cast
from ftplib import error_perm, error_temp
from six import PY2, raise_from, text_type
from . import _ftp_parse as ftp_parse
from . import errors
from .base import FS
from .constants import DEFAULT_CHUNK_SIZE
from .enums import ResourceType, Seek
from .info import Info
from .iotools import line_iterator
from .mode import Mode
from .path import abspath, basename, dirname, normpath, split
from .time import epoch_to_datetime
class FTPFS(FS):
    """A FTP (File Transport Protocol) Filesystem.

    Optionally, the connection can be made securely via TLS. This is known as
    FTPS, or FTP Secure. TLS will be enabled when using the ftps:// protocol,
    or when setting the `tls` argument to True in the constructor.

    Examples:
        Create with the constructor::

            >>> from fs.ftpfs import FTPFS
            >>> ftp_fs = FTPFS("demo.wftpserver.com")

        Or via an FS URL::

            >>> ftp_fs = fs.open_fs('ftp://test.rebex.net')

        Or via an FS URL, using TLS::

            >>> ftp_fs = fs.open_fs('ftps://demo.wftpserver.com')

        You can also use a non-anonymous username, and optionally a
        password, even within a FS URL::

            >>> ftp_fs = FTPFS("test.rebex.net", user="demo", passwd="password")
            >>> ftp_fs = fs.open_fs('ftp://demo:password@test.rebex.net')

        Connecting via a proxy is supported. If using a FS URL, the proxy
        URL will need to be added as a URL parameter::

            >>> ftp_fs = FTPFS("ftp.ebi.ac.uk", proxy="test.rebex.net")
            >>> ftp_fs = fs.open_fs('ftp://ftp.ebi.ac.uk/?proxy=test.rebex.net')

    """
    _meta = {'invalid_path_chars': '\x00', 'network': True, 'read_only': False, 'thread_safe': True, 'unicode_paths': True, 'virtual': False}

    def __init__(self, host, user='anonymous', passwd='', acct='', timeout=10, port=21, proxy=None, tls=False):
        """Create a new `FTPFS` instance.

        Arguments:
            host (str): A FTP host, e.g. ``'ftp.mirror.nl'``.
            user (str): A username (default is ``'anonymous'``).
            passwd (str): Password for the server, or `None` for anon.
            acct (str): FTP account.
            timeout (int): Timeout for contacting server (in seconds,
                defaults to 10).
            port (int): FTP port number (default 21).
            proxy (str, optional): An FTP proxy, or ``None`` (default)
                for no proxy.
            tls (bool): Attempt to use FTP over TLS (FTPS) (default: False)

        """
        super(FTPFS, self).__init__()
        self._host = host
        self._user = user
        self.passwd = passwd
        self.acct = acct
        self.timeout = timeout
        self.port = port
        self.proxy = proxy
        self.tls = tls
        if self.tls and isinstance(FTP_TLS, Exception):
            raise_from(errors.CreateFailed('FTP over TLS not supported'), FTP_TLS)
        self.encoding = 'latin-1'
        self._ftp = None
        self._welcome = None
        self._features = {}

    def __repr__(self):
        return 'FTPFS({!r}, port={!r})'.format(self.host, self.port)

    def __str__(self):
        _fmt = "<ftpfs '{host}'>" if self.port == 21 else "<ftpfs '{host}:{port}'>"
        return _fmt.format(host=self.host, port=self.port)

    @property
    def user(self):
        return self._user if self.proxy is None else '{}@{}'.format(self._user, self._host)

    @property
    def host(self):
        return self._host if self.proxy is None else self.proxy

    @classmethod
    def _parse_features(cls, feat_response):
        """Parse a dict of features from FTP feat response."""
        features = {}
        if feat_response.split('-')[0] == '211':
            for line in feat_response.splitlines():
                if line.startswith(' '):
                    key, _, value = line[1:].partition(' ')
                    features[key] = value
        return features

    def _open_ftp(self):
        """Open a new ftp object."""
        _ftp = FTP_TLS() if self.tls else FTP()
        _ftp.set_debuglevel(0)
        with ftp_errors(self):
            _ftp.connect(self.host, self.port, self.timeout)
            _ftp.login(self.user, self.passwd, self.acct)
            try:
                _ftp.prot_p()
            except AttributeError:
                pass
            self._features = {}
            try:
                feat_response = _decode(_ftp.sendcmd('FEAT'), 'latin-1')
            except error_perm:
                self.encoding = 'latin-1'
            else:
                self._features = self._parse_features(feat_response)
                self.encoding = 'utf-8' if 'UTF8' in self._features else 'latin-1'
                if not PY2:
                    _ftp.file = _ftp.sock.makefile('r', encoding=self.encoding)
        _ftp.encoding = self.encoding
        self._welcome = _ftp.welcome
        return _ftp

    def _manage_ftp(self):
        ftp = self._open_ftp()
        return manage_ftp(ftp)

    @property
    def ftp_url(self):
        """Get the FTP url this filesystem will open."""
        if self.port == 21:
            _host_part = self.host
        else:
            _host_part = '{}:{}'.format(self.host, self.port)
        if self.user == 'anonymous' or self.user is None:
            _user_part = ''
        else:
            _user_part = '{}:{}@'.format(self.user, self.passwd)
        scheme = 'ftps' if self.tls else 'ftp'
        url = '{}://{}{}'.format(scheme, _user_part, _host_part)
        return url

    @property
    def ftp(self):
        """~ftplib.FTP: the underlying FTP client."""
        return self._get_ftp()

    def geturl(self, path, purpose='download'):
        """Get FTP url for resource."""
        _path = self.validatepath(path)
        if purpose != 'download':
            raise errors.NoURL(_path, purpose)
        return '{}{}'.format(self.ftp_url, _path)

    def _get_ftp(self):
        if self._ftp is None:
            self._ftp = self._open_ftp()
        return self._ftp

    @property
    def features(self):
        """`dict`: Features of the remote FTP server."""
        self._get_ftp()
        return self._features

    def _read_dir(self, path):
        _path = abspath(normpath(path))
        lines = []
        with ftp_errors(self, path=path):
            self.ftp.retrlines(str('LIST ') + _encode(_path, self.ftp.encoding), lines.append)
        lines = [line.decode('utf-8') if isinstance(line, bytes) else line for line in lines]
        _list = [Info(raw_info) for raw_info in ftp_parse.parse(lines)]
        dir_listing = OrderedDict({info.name: info for info in _list})
        return dir_listing

    @property
    def supports_mlst(self):
        """bool: whether the server supports MLST feature."""
        return 'MLST' in self.features

    @property
    def supports_mdtm(self):
        """bool: whether the server supports the MDTM feature."""
        return 'MDTM' in self.features

    def create(self, path, wipe=False):
        _path = self.validatepath(path)
        with ftp_errors(self, path):
            if wipe or not self.isfile(path):
                empty_file = io.BytesIO()
                self.ftp.storbinary(str('STOR ') + _encode(_path, self.ftp.encoding), empty_file)
                return True
        return False

    @classmethod
    def _parse_ftp_time(cls, time_text):
        """Parse a time from an ftp directory listing."""
        try:
            tm_year = int(time_text[0:4])
            tm_month = int(time_text[4:6])
            tm_day = int(time_text[6:8])
            tm_hour = int(time_text[8:10])
            tm_min = int(time_text[10:12])
            tm_sec = int(time_text[12:14])
        except ValueError:
            return None
        epoch_time = calendar.timegm((tm_year, tm_month, tm_day, tm_hour, tm_min, tm_sec))
        return epoch_time

    @classmethod
    def _parse_facts(cls, line):
        name = None
        facts = {}
        for fact in line.split(';'):
            key, sep, value = fact.partition('=')
            if sep:
                key = key.strip().lower()
                value = value.strip()
                facts[key] = value
            else:
                name = basename(fact.rstrip('/').strip())
        return (name if name not in ('.', '..') else None, facts)

    @classmethod
    def _parse_mlsx(cls, lines):
        for line in lines:
            name, facts = cls._parse_facts(line.strip())
            if name is None:
                continue
            _type = facts.get('type', 'file')
            if _type not in {'dir', 'file'}:
                continue
            is_dir = _type == 'dir'
            raw_info = {}
            raw_info['basic'] = {'name': name, 'is_dir': is_dir}
            raw_info['ftp'] = facts
            raw_info['details'] = {'type': int(ResourceType.directory if is_dir else ResourceType.file)}
            details = raw_info['details']
            size_str = facts.get('size', facts.get('sizd', '0'))
            size = 0
            if size_str.isdigit():
                size = int(size_str)
            details['size'] = size
            if 'modify' in facts:
                details['modified'] = cls._parse_ftp_time(facts['modify'])
            if 'create' in facts:
                details['created'] = cls._parse_ftp_time(facts['create'])
            yield raw_info
    if typing.TYPE_CHECKING:

        def opendir(self, path, factory=None):
            pass

    def getinfo(self, path, namespaces=None):
        _path = self.validatepath(path)
        namespaces = namespaces or ()
        if _path == '/':
            return Info({'basic': {'name': '', 'is_dir': True}, 'details': {'type': int(ResourceType.directory)}})
        if self.supports_mlst:
            with self._lock:
                with ftp_errors(self, path=path):
                    response = self.ftp.sendcmd(str('MLST ') + _encode(_path, self.ftp.encoding))
                lines = _decode(response, self.ftp.encoding).splitlines()[1:-1]
                for raw_info in self._parse_mlsx(lines):
                    return Info(raw_info)
        with ftp_errors(self, path=path):
            dir_name, file_name = split(_path)
            directory = self._read_dir(dir_name)
            if file_name not in directory:
                raise errors.ResourceNotFound(path)
            info = directory[file_name]
            return info

    def getmeta(self, namespace='standard'):
        _meta = {}
        self._get_ftp()
        if namespace == 'standard':
            _meta = self._meta.copy()
            _meta['unicode_paths'] = 'UTF8' in self.features
            _meta['supports_mtime'] = 'MDTM' in self.features
        return _meta

    def getmodified(self, path):
        if self.supports_mdtm:
            _path = self.validatepath(path)
            with self._lock:
                with ftp_errors(self, path=path):
                    cmd = 'MDTM ' + _encode(_path, self.ftp.encoding)
                    response = self.ftp.sendcmd(cmd)
                    mtime = self._parse_ftp_time(response.split()[1])
                    return epoch_to_datetime(mtime)
        return super(FTPFS, self).getmodified(path)

    def listdir(self, path):
        _path = self.validatepath(path)
        with self._lock:
            dir_list = [info.name for info in self.scandir(_path)]
        return dir_list

    def makedir(self, path, permissions=None, recreate=False):
        _path = self.validatepath(path)
        with ftp_errors(self, path=path):
            if _path == '/':
                if recreate:
                    return self.opendir(path)
                else:
                    raise errors.DirectoryExists(path)
            if not (recreate and self.isdir(path)):
                try:
                    self.ftp.mkd(_encode(_path, self.ftp.encoding))
                except error_perm as error:
                    code, _ = _parse_ftp_error(error)
                    if code == '550':
                        if self.isdir(path):
                            raise errors.DirectoryExists(path)
                        elif self.exists(path):
                            raise errors.DirectoryExists(path)
                    raise errors.ResourceNotFound(path)
        return self.opendir(path)

    def openbin(self, path, mode='r', buffering=-1, **options):
        _mode = Mode(mode)
        _mode.validate_bin()
        _path = self.validatepath(path)
        with self._lock:
            try:
                info = self.getinfo(_path)
            except errors.ResourceNotFound:
                if _mode.reading:
                    raise errors.ResourceNotFound(path)
                if _mode.writing and (not self.isdir(dirname(_path))):
                    raise errors.ResourceNotFound(path)
            else:
                if info.is_dir:
                    raise errors.FileExpected(path)
                if _mode.exclusive:
                    raise errors.FileExists(path)
            ftp_file = FTPFile(self, _path, _mode.to_platform_bin())
        return ftp_file

    def remove(self, path):
        self.check()
        _path = self.validatepath(path)
        with self._lock:
            if self.isdir(path):
                raise errors.FileExpected(path=path)
            with ftp_errors(self, path):
                self.ftp.delete(_encode(_path, self.ftp.encoding))

    def removedir(self, path):
        _path = self.validatepath(path)
        if _path == '/':
            raise errors.RemoveRootError()
        with ftp_errors(self, path):
            try:
                self.ftp.rmd(_encode(_path, self.ftp.encoding))
            except error_perm as error:
                code, _ = _parse_ftp_error(error)
                if code == '550':
                    if self.isfile(path):
                        raise errors.DirectoryExpected(path)
                    if not self.isempty(path):
                        raise errors.DirectoryNotEmpty(path)
                raise

    def _scandir(self, path, namespaces=None):
        _path = self.validatepath(path)
        with self._lock:
            if self.supports_mlst:
                lines = []
                with ftp_errors(self, path=path):
                    try:
                        self.ftp.retrlines(str('MLSD ') + _encode(_path, self.ftp.encoding), lambda l: lines.append(_decode(l, self.ftp.encoding)))
                    except error_perm:
                        if not self.getinfo(path).is_dir:
                            raise errors.DirectoryExpected(path)
                        raise
                if lines:
                    for raw_info in self._parse_mlsx(lines):
                        yield Info(raw_info)
                    return
            for info in self._read_dir(_path).values():
                yield info

    def scandir(self, path, namespaces=None, page=None):
        if not self.supports_mlst and (not self.getinfo(path).is_dir):
            raise errors.DirectoryExpected(path)
        iter_info = self._scandir(path, namespaces=namespaces)
        if page is not None:
            start, end = page
            iter_info = itertools.islice(iter_info, start, end)
        return iter_info

    def upload(self, path, file, chunk_size=None, **options):
        _path = self.validatepath(path)
        with self._lock:
            with ftp_errors(self, path):
                self.ftp.storbinary(str('STOR ') + _encode(_path, self.ftp.encoding), file)

    def writebytes(self, path, contents):
        if not isinstance(contents, bytes):
            raise TypeError('contents must be bytes')
        self.upload(path, io.BytesIO(contents))

    def setinfo(self, path, info):
        use_mfmt = False
        if 'MFMT' in self.features:
            info_details = None
            if 'modified' in info:
                info_details = info['modified']
            elif 'details' in info:
                info_details = info['details']
            if info_details and 'modified' in info_details:
                use_mfmt = True
                mtime = cast(float, info_details['modified'])
        if use_mfmt:
            with ftp_errors(self, path):
                cmd = 'MFMT ' + datetime.datetime.utcfromtimestamp(mtime).strftime('%Y%m%d%H%M%S') + ' ' + _encode(path, self.ftp.encoding)
                try:
                    self.ftp.sendcmd(cmd)
                except error_perm:
                    pass
        elif not self.exists(path):
            raise errors.ResourceNotFound(path)

    def readbytes(self, path):
        _path = self.validatepath(path)
        data = io.BytesIO()
        with ftp_errors(self, path):
            with self._manage_ftp() as ftp:
                try:
                    ftp.retrbinary(str('RETR ') + _encode(_path, self.ftp.encoding), data.write)
                except error_perm as error:
                    code, _ = _parse_ftp_error(error)
                    if code == '550':
                        if self.isdir(path):
                            raise errors.FileExpected(path)
                    raise
        data_bytes = data.getvalue()
        return data_bytes

    def close(self):
        if not self.isclosed():
            try:
                self.ftp.quit()
            except Exception:
                pass
            self._ftp = None
        super(FTPFS, self).close()