import bisect
import errno
import itertools
import os
import random
import stat
import sys
import time
import warnings
from .. import config, debug, errors, urlutils
from ..errors import LockError, ParamikoNotPresent, PathError, TransportError
from ..osutils import fancy_rename
from ..trace import mutter, warning
from ..transport import (ConnectedTransport, FileExists, FileFileStream,
class SFTPTransport(ConnectedTransport):
    """Transport implementation for SFTP access."""
    _max_readv_combine = 200
    _bytes_to_read_before_seek = 8192
    _max_request_size = 32768

    def _remote_path(self, relpath):
        """Return the path to be passed along the sftp protocol for relpath.

        :param relpath: is a urlencoded string.
        """
        remote_path = self._parsed_url.clone(relpath).path
        if remote_path.startswith('/~/'):
            remote_path = remote_path[3:]
        elif remote_path == '/~':
            remote_path = ''
        return remote_path

    def _create_connection(self, credentials=None):
        """Create a new connection with the provided credentials.

        :param credentials: The credentials needed to establish the connection.

        :return: The created connection and its associated credentials.

        The credentials are only the password as it may have been entered
        interactively by the user and may be different from the one provided
        in base url at transport creation time.
        """
        if credentials is None:
            password = self._parsed_url.password
        else:
            password = credentials
        vendor = ssh._get_ssh_vendor()
        user = self._parsed_url.user
        if user is None:
            auth = config.AuthenticationConfig()
            user = auth.get_user('ssh', self._parsed_url.host, self._parsed_url.port)
        connection = vendor.connect_sftp(self._parsed_url.user, password, self._parsed_url.host, self._parsed_url.port)
        return (connection, (user, password))

    def disconnect(self):
        connection = self._get_connection()
        if connection is not None:
            connection.close()

    def _get_sftp(self):
        """Ensures that a connection is established"""
        connection = self._get_connection()
        if connection is None:
            connection, credentials = self._create_connection()
            self._set_connection(connection, credentials)
        return connection

    def has(self, relpath):
        """
        Does the target location exist?
        """
        try:
            self._get_sftp().stat(self._remote_path(relpath))
            self._report_activity(20, 'read')
            return True
        except OSError:
            return False

    def get(self, relpath):
        """Get the file at the given relative path.

        :param relpath: The relative path to the file
        """
        try:
            path = self._remote_path(relpath)
            f = self._get_sftp().file(path, mode='rb')
            size = f.stat().st_size
            if getattr(f, 'prefetch', None) is not None:
                f.prefetch(size)
            return f
        except (OSError, paramiko.SSHException) as e:
            self._translate_io_exception(e, path, ': error retrieving', failure_exc=errors.ReadError)

    def get_bytes(self, relpath):
        with self.get(relpath) as f:
            bytes = f.read()
            self._report_activity(len(bytes), 'read')
            return bytes

    def _readv(self, relpath, offsets):
        """See Transport.readv()"""
        if not offsets:
            return
        try:
            path = self._remote_path(relpath)
            fp = self._get_sftp().file(path, mode='rb')
            readv = getattr(fp, 'readv', None)
            if readv:
                return self._sftp_readv(fp, offsets, relpath)
            if 'sftp' in debug.debug_flags:
                mutter('seek and read %s offsets', len(offsets))
            return self._seek_and_read(fp, offsets, relpath)
        except (OSError, paramiko.SSHException) as e:
            self._translate_io_exception(e, path, ': error retrieving')

    def recommended_page_size(self):
        """See Transport.recommended_page_size().

        For SFTP we suggest a large page size to reduce the overhead
        introduced by latency.
        """
        return 64 * 1024

    def _sftp_readv(self, fp, offsets, relpath):
        """Use the readv() member of fp to do async readv.

        Then read them using paramiko.readv(). paramiko.readv()
        does not support ranges > 64K, so it caps the request size, and
        just reads until it gets all the stuff it wants.
        """
        helper = _SFTPReadvHelper(offsets, relpath, self._report_activity)
        return helper.request_and_yield_offsets(fp)

    def put_file(self, relpath, f, mode=None):
        """
        Copy the file-like object into the location.

        :param relpath: Location to put the contents, relative to base.
        :param f:       File-like object.
        :param mode: The final mode for the file
        """
        final_path = self._remote_path(relpath)
        return self._put(final_path, f, mode=mode)

    def _put(self, abspath, f, mode=None):
        """Helper function so both put() and copy_abspaths can reuse the code"""
        tmp_abspath = '%s.tmp.%.9f.%d.%d' % (abspath, time.time(), os.getpid(), random.randint(0, 2147483647))
        fout = self._sftp_open_exclusive(tmp_abspath, mode=mode)
        closed = False
        try:
            try:
                fout.set_pipelined(True)
                length = self._pump(f, fout)
            except (OSError, paramiko.SSHException) as e:
                self._translate_io_exception(e, tmp_abspath)
            if mode is not None:
                self._get_sftp().chmod(tmp_abspath, mode)
            fout.close()
            closed = True
            self._rename_and_overwrite(tmp_abspath, abspath)
            return length
        except Exception as e:
            import traceback
            mutter(traceback.format_exc())
            try:
                if not closed:
                    fout.close()
                self._get_sftp().remove(tmp_abspath)
            except:
                raise e
            raise

    def _put_non_atomic_helper(self, relpath, writer, mode=None, create_parent_dir=False, dir_mode=None):
        abspath = self._remote_path(relpath)

        def _open_and_write_file():
            """Try to open the target file, raise error on failure"""
            fout = None
            try:
                try:
                    fout = self._get_sftp().file(abspath, mode='wb')
                    fout.set_pipelined(True)
                    writer(fout)
                except (paramiko.SSHException, OSError) as e:
                    self._translate_io_exception(e, abspath, ': unable to open')
                if mode is not None:
                    self._get_sftp().chmod(abspath, mode)
            finally:
                if fout is not None:
                    fout.close()
        if not create_parent_dir:
            _open_and_write_file()
            return
        try:
            _open_and_write_file()
        except NoSuchFile:
            parent_dir = os.path.dirname(abspath)
            self._mkdir(parent_dir, dir_mode)
            _open_and_write_file()

    def put_file_non_atomic(self, relpath, f, mode=None, create_parent_dir=False, dir_mode=None):
        """Copy the file-like object into the target location.

        This function is not strictly safe to use. It is only meant to
        be used when you already know that the target does not exist.
        It is not safe, because it will open and truncate the remote
        file. So there may be a time when the file has invalid contents.

        :param relpath: The remote location to put the contents.
        :param f:       File-like object.
        :param mode:    Possible access permissions for new file.
                        None means do not set remote permissions.
        :param create_parent_dir: If we cannot create the target file because
                        the parent directory does not exist, go ahead and
                        create it, and then try again.
        """

        def writer(fout):
            self._pump(f, fout)
        self._put_non_atomic_helper(relpath, writer, mode=mode, create_parent_dir=create_parent_dir, dir_mode=dir_mode)

    def put_bytes_non_atomic(self, relpath: str, raw_bytes: bytes, mode=None, create_parent_dir=False, dir_mode=None):
        if not isinstance(raw_bytes, bytes):
            raise TypeError('raw_bytes must be a plain string, not %s' % type(raw_bytes))

        def writer(fout):
            fout.write(raw_bytes)
        self._put_non_atomic_helper(relpath, writer, mode=mode, create_parent_dir=create_parent_dir, dir_mode=dir_mode)

    def iter_files_recursive(self):
        """Walk the relative paths of all files in this transport."""
        queue = list(self.list_dir('.'))
        while queue:
            relpath = queue.pop(0)
            st = self.stat(relpath)
            if stat.S_ISDIR(st.st_mode):
                for i, basename in enumerate(self.list_dir(relpath)):
                    queue.insert(i, relpath + '/' + basename)
            else:
                yield relpath

    def _mkdir(self, abspath, mode=None):
        if mode is None:
            local_mode = 511
        else:
            local_mode = mode
        try:
            self._report_activity(len(abspath), 'write')
            self._get_sftp().mkdir(abspath, local_mode)
            self._report_activity(1, 'read')
            if mode is not None:
                stat = self._get_sftp().lstat(abspath)
                mode = mode & 511
                if mode != stat.st_mode & 511:
                    if stat.st_mode & 3072:
                        warning('About to chmod %s over sftp, which will result in its suid or sgid bits being cleared.  If you want to preserve those bits, change your  environment on the server to use umask 0%03o.' % (abspath, 511 - mode))
                    self._get_sftp().chmod(abspath, mode=mode)
        except (paramiko.SSHException, OSError) as e:
            self._translate_io_exception(e, abspath, ': unable to mkdir', failure_exc=FileExists)

    def mkdir(self, relpath, mode=None):
        """Create a directory at the given path."""
        self._mkdir(self._remote_path(relpath), mode=mode)

    def open_write_stream(self, relpath, mode=None):
        """See Transport.open_write_stream."""
        self.put_bytes_non_atomic(relpath, b'', mode)
        abspath = self._remote_path(relpath)
        handle = None
        try:
            handle = self._get_sftp().file(abspath, mode='wb')
            handle.set_pipelined(True)
        except (paramiko.SSHException, OSError) as e:
            self._translate_io_exception(e, abspath, ': unable to open')
        _file_streams[self.abspath(relpath)] = handle
        return FileFileStream(self, relpath, handle)

    def _translate_io_exception(self, e, path, more_info='', failure_exc=PathError):
        """Translate a paramiko or IOError into a friendlier exception.

        :param e: The original exception
        :param path: The path in question when the error is raised
        :param more_info: Extra information that can be included,
                          such as what was going on
        :param failure_exc: Paramiko has the super fun ability to raise completely
                           opaque errors that just set "e.args = ('Failure',)" with
                           no more information.
                           If this parameter is set, it defines the exception
                           to raise in these cases.
        """
        self._translate_error(e, path, raise_generic=False)
        if getattr(e, 'args', None) is not None:
            if e.args == ('No such file or directory',) or e.args == ('No such file',):
                raise NoSuchFile(path, str(e) + more_info)
            if e.args == ('mkdir failed',) or e.args[0].startswith('syserr: File exists'):
                raise FileExists(path, str(e) + more_info)
            if e.args == ('Failure',):
                raise failure_exc(path, str(e) + more_info)
            if e.args[0].startswith('Directory not empty: ') or getattr(e, 'errno', None) == errno.ENOTEMPTY:
                raise errors.DirectoryNotEmpty(path, str(e))
            if e.args == ('Operation unsupported',):
                raise errors.TransportNotPossible()
            mutter('Raising exception with args %s', e.args)
        if getattr(e, 'errno', None) is not None:
            mutter('Raising exception with errno %s', e.errno)
        raise e

    def append_file(self, relpath, f, mode=None):
        """
        Append the text in the file-like object into the final
        location.
        """
        try:
            path = self._remote_path(relpath)
            fout = self._get_sftp().file(path, 'ab')
            if mode is not None:
                self._get_sftp().chmod(path, mode)
            result = fout.tell()
            self._pump(f, fout)
            return result
        except (OSError, paramiko.SSHException) as e:
            self._translate_io_exception(e, relpath, ': unable to append')

    def rename(self, rel_from, rel_to):
        """Rename without special overwriting"""
        try:
            self._get_sftp().rename(self._remote_path(rel_from), self._remote_path(rel_to))
        except (OSError, paramiko.SSHException) as e:
            self._translate_io_exception(e, rel_from, ': unable to rename to %r' % rel_to)

    def _rename_and_overwrite(self, abs_from, abs_to):
        """Do a fancy rename on the remote server.

        Using the implementation provided by osutils.
        """
        try:
            sftp = self._get_sftp()
            fancy_rename(abs_from, abs_to, rename_func=sftp.rename, unlink_func=sftp.remove)
        except (OSError, paramiko.SSHException) as e:
            self._translate_io_exception(e, abs_from, ': unable to rename to %r' % abs_to)

    def move(self, rel_from, rel_to):
        """Move the item at rel_from to the location at rel_to"""
        path_from = self._remote_path(rel_from)
        path_to = self._remote_path(rel_to)
        self._rename_and_overwrite(path_from, path_to)

    def delete(self, relpath):
        """Delete the item at relpath"""
        path = self._remote_path(relpath)
        try:
            self._get_sftp().remove(path)
        except (OSError, paramiko.SSHException) as e:
            self._translate_io_exception(e, path, ': unable to delete')

    def external_url(self):
        """See breezy.transport.Transport.external_url."""
        return self.base

    def listable(self):
        """Return True if this store supports listing."""
        return True

    def list_dir(self, relpath):
        """
        Return a list of all files at the given location.
        """
        path = self._remote_path(relpath)
        try:
            entries = self._get_sftp().listdir(path)
            self._report_activity(sum(map(len, entries)), 'read')
        except (OSError, paramiko.SSHException) as e:
            self._translate_io_exception(e, path, ': failed to list_dir')
        return [urlutils.escape(entry) for entry in entries]

    def rmdir(self, relpath):
        """See Transport.rmdir."""
        path = self._remote_path(relpath)
        try:
            return self._get_sftp().rmdir(path)
        except (OSError, paramiko.SSHException) as e:
            self._translate_io_exception(e, path, ': failed to rmdir')

    def stat(self, relpath):
        """Return the stat information for a file."""
        path = self._remote_path(relpath)
        try:
            return self._get_sftp().lstat(path)
        except (OSError, paramiko.SSHException) as e:
            self._translate_io_exception(e, path, ': unable to stat')

    def readlink(self, relpath):
        """See Transport.readlink."""
        path = self._remote_path(relpath)
        try:
            return self._get_sftp().readlink(self._remote_path(path))
        except (OSError, paramiko.SSHException) as e:
            self._translate_io_exception(e, path, ': unable to readlink')

    def symlink(self, source, link_name):
        """See Transport.symlink."""
        try:
            conn = self._get_sftp()
            sftp_retval = conn.symlink(source, self._remote_path(link_name))
        except (OSError, paramiko.SSHException) as e:
            self._translate_io_exception(e, link_name, ': unable to create symlink to %r' % source)

    def lock_read(self, relpath):
        """
        Lock the given file for shared (read) access.
        :return: A lock object, which has an unlock() member function
        """

        class BogusLock:

            def __init__(self, path):
                self.path = path

            def unlock(self):
                pass

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def __enter__(self):
                pass
        return BogusLock(relpath)

    def lock_write(self, relpath):
        """
        Lock the given file for exclusive (write) access.
        WARNING: many transports do not support this, so trying avoid using it

        :return: A lock object, which has an unlock() member function
        """
        return SFTPLock(relpath, self)

    def _sftp_open_exclusive(self, abspath, mode=None):
        """Open a remote path exclusively.

        SFTP supports O_EXCL (SFTP_FLAG_EXCL), which fails if
        the file already exists. However it does not expose this
        at the higher level of SFTPClient.open(), so we have to
        sneak away with it.

        WARNING: This breaks the SFTPClient abstraction, so it
        could easily break against an updated version of paramiko.

        :param abspath: The remote absolute path where the file should be opened
        :param mode: The mode permissions bits for the new file
        """
        path = self._get_sftp()._adjust_cwd(abspath)
        attr = SFTPAttributes()
        if mode is not None:
            attr.st_mode = mode
        omode = SFTP_FLAG_WRITE | SFTP_FLAG_CREATE | SFTP_FLAG_TRUNC | SFTP_FLAG_EXCL
        try:
            t, msg = self._get_sftp()._request(CMD_OPEN, path, omode, attr)
            if t != CMD_HANDLE:
                raise TransportError('Expected an SFTP handle')
            handle = msg.get_string()
            return SFTPFile(self._get_sftp(), handle, 'wb', -1)
        except (paramiko.SSHException, OSError) as e:
            self._translate_io_exception(e, abspath, ': unable to open', failure_exc=FileExists)

    def _can_roundtrip_unix_modebits(self):
        if sys.platform == 'win32':
            return False
        else:
            return True