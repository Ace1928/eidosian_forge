import os
import random
import stat
import time
from io import BytesIO
from urllib.parse import urlparse, urlunparse
from .. import config, debug, errors, osutils, ui, urlutils
from ..tests.test_server import TestServer
from ..trace import mutter
from . import (ConnectedTransport, FileExists, FileStream, NoSuchFile,
class GioTransport(ConnectedTransport):
    """This is the transport agent for gio+XXX:// access."""

    def __init__(self, base, _from_transport=None):
        """Initialize the GIO transport and make sure the url is correct."""
        if not base.startswith('gio+'):
            raise ValueError(base)
        scheme, netloc, path, params, query, fragment = urlparse(base[len('gio+'):], allow_fragments=False)
        if '@' in netloc:
            user, netloc = netloc.rsplit('@', 1)
        gio_backends = ['dav', 'file', 'ftp', 'obex', 'sftp', 'ssh', 'smb']
        if scheme not in gio_backends:
            raise urlutils.InvalidURL(base, extra='GIO support is only available for ' + ', '.join(gio_backends))
        u = (scheme, netloc, path, '', '', '')
        self.url = urlunparse(u)
        super().__init__(base, _from_transport=_from_transport)

    def _relpath_to_url(self, relpath):
        full_url = urlutils.join(self.url, relpath)
        if isinstance(full_url, str):
            raise urlutils.InvalidURL(full_url)
        return full_url

    def _get_GIO(self, relpath):
        """Return the ftplib.GIO instance for this object."""
        connection = self._get_connection()
        if connection is None:
            connection, credentials = self._create_connection()
            self._set_connection(connection, credentials)
        fileurl = self._relpath_to_url(relpath)
        file = gio.File(fileurl)
        return file

    def _auth_cb(self, op, message, default_user, default_domain, flags):
        auth = config.AuthenticationConfig()
        parsed_url = urlutils.URL.from_string(self.url)
        user = None
        if flags & gio.ASK_PASSWORD_NEED_USERNAME and flags & gio.ASK_PASSWORD_NEED_DOMAIN:
            prompt = '{}'.format(parsed_url.scheme.upper()) + ' %(host)s DOMAIN\\username'
            user_and_domain = auth.get_user(parsed_url.scheme, parsed_url.host, port=parsed_url.port, ask=True, prompt=prompt)
            domain, user = user_and_domain.split('\\', 1)
            op.set_username(user)
            op.set_domain(domain)
        elif flags & gio.ASK_PASSWORD_NEED_USERNAME:
            user = auth.get_user(parsed_url.scheme, parsed_url.host, port=parsed_url.port, ask=True)
            op.set_username(user)
        elif flags & gio.ASK_PASSWORD_NEED_DOMAIN:
            prompt = '{}'.format(parsed_url.scheme.upper()) + ' %(host)s DOMAIN'
            domain = ui.ui_factory.get_username(prompt=prompt)
            op.set_domain(domain)
        if flags & gio.ASK_PASSWORD_NEED_PASSWORD:
            if user is None:
                user = op.get_username()
            password = auth.get_password(parsed_url.scheme, parsed_url.host, user, port=parsed_url.port)
            op.set_password(password)
        op.reply(gio.MOUNT_OPERATION_HANDLED)

    def _mount_done_cb(self, obj, res):
        try:
            obj.mount_enclosing_volume_finish(res)
            self.loop.quit()
        except gio.Error as e:
            self.loop.quit()
            raise errors.BzrError('Failed to mount the given location: ' + str(e))

    def _create_connection(self, credentials=None):
        if credentials is None:
            user, password = (self._parsed_url.user, self._parsed_url.password)
        else:
            user, password = credentials
        try:
            connection = gio.File(self.url)
            mount = None
            try:
                mount = connection.find_enclosing_mount()
            except gio.Error as e:
                if e.code == gio.ERROR_NOT_MOUNTED:
                    self.loop = glib.MainLoop()
                    ui.ui_factory.show_message('Mounting %s using GIO' % self.url)
                    op = gio.MountOperation()
                    if user:
                        op.set_username(user)
                    if password:
                        op.set_password(password)
                    op.connect('ask-password', self._auth_cb)
                    m = connection.mount_enclosing_volume(op, self._mount_done_cb)
                    self.loop.run()
        except gio.Error as e:
            raise errors.TransportError(msg='Error setting up connection: %s' % str(e), orig_error=e)
        return (connection, (user, password))

    def disconnect(self):
        pass

    def _reconnect(self):
        """Create a new connection with the previously used credentials"""
        credentials = self._get_credentials()
        connection, credentials = self._create_connection(credentials)
        self._set_connection(connection, credentials)

    def _remote_path(self, relpath):
        return self._parsed_url.clone(relpath).path

    def has(self, relpath):
        """Does the target location exist?"""
        try:
            if 'gio' in debug.debug_flags:
                mutter('GIO has check: %s' % relpath)
            f = self._get_GIO(relpath)
            st = GioStatResult(f)
            if stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode):
                return True
            return False
        except gio.Error as e:
            if e.code == gio.ERROR_NOT_FOUND:
                return False
            else:
                self._translate_gio_error(e, relpath)

    def get(self, relpath, retries=0):
        """Get the file at the given relative path.

        :param relpath: The relative path to the file
        :param retries: Number of retries after temporary failures so far
                        for this operation.

        We're meant to return a file-like object which bzr will
        then read from. For now we do this via the magic of BytesIO
        """
        try:
            if 'gio' in debug.debug_flags:
                mutter('GIO get: %s' % relpath)
            f = self._get_GIO(relpath)
            fin = f.read()
            buf = fin.read()
            fin.close()
            return BytesIO(buf)
        except gio.Error as e:
            if e.code == gio.ERROR_NOT_MOUNTED:
                raise errors.PathError(relpath, extra='Failed to get file, make sure the path is correct. ' + str(e))
            else:
                self._translate_gio_error(e, relpath)

    def put_file(self, relpath, fp, mode=None):
        """Copy the file-like object into the location.

        :param relpath: Location to put the contents, relative to base.
        :param fp:       File-like or string object.
        """
        if 'gio' in debug.debug_flags:
            mutter('GIO put_file %s' % relpath)
        tmppath = '%s.tmp.%.9f.%d.%d' % (relpath, time.time(), os.getpid(), random.randint(0, 2147483647))
        f = None
        fout = None
        try:
            closed = True
            try:
                f = self._get_GIO(tmppath)
                fout = f.create()
                closed = False
                length = self._pump(fp, fout)
                fout.close()
                closed = True
                self.stat(tmppath)
                dest = self._get_GIO(relpath)
                f.move(dest, flags=gio.FILE_COPY_OVERWRITE)
                f = None
                if mode is not None:
                    self._setmode(relpath, mode)
                return length
            except gio.Error as e:
                self._translate_gio_error(e, relpath)
        finally:
            if not closed and fout is not None:
                fout.close()
            if f is not None and f.query_exists():
                f.delete()

    def mkdir(self, relpath, mode=None):
        """Create a directory at the given path."""
        try:
            if 'gio' in debug.debug_flags:
                mutter('GIO mkdir: %s' % relpath)
            f = self._get_GIO(relpath)
            f.make_directory()
            self._setmode(relpath, mode)
        except gio.Error as e:
            self._translate_gio_error(e, relpath)

    def open_write_stream(self, relpath, mode=None):
        """See Transport.open_write_stream."""
        if 'gio' in debug.debug_flags:
            mutter('GIO open_write_stream %s' % relpath)
        if mode is not None:
            self._setmode(relpath, mode)
        result = GioFileStream(self, relpath)
        _file_streams[self.abspath(relpath)] = result
        return result

    def recommended_page_size(self):
        """See Transport.recommended_page_size().

        For FTP we suggest a large page size to reduce the overhead
        introduced by latency.
        """
        if 'gio' in debug.debug_flags:
            mutter('GIO recommended_page')
        return 64 * 1024

    def rmdir(self, relpath):
        """Delete the directory at rel_path"""
        try:
            if 'gio' in debug.debug_flags:
                mutter('GIO rmdir %s' % relpath)
            st = self.stat(relpath)
            if stat.S_ISDIR(st.st_mode):
                f = self._get_GIO(relpath)
                f.delete()
            else:
                raise errors.NotADirectory(relpath)
        except gio.Error as e:
            self._translate_gio_error(e, relpath)
        except errors.NotADirectory as e:
            raise e
        except Exception as e:
            mutter('failed to rmdir {}: {}'.format(relpath, e))
            raise errors.PathError(relpath)

    def append_file(self, relpath, file, mode=None):
        """Append the text in the file-like object into the final
        location.
        """
        if 'gio' in debug.debug_flags:
            mutter('GIO append_file: %s' % relpath)
        tmppath = '%s.tmp.%.9f.%d.%d' % (relpath, time.time(), os.getpid(), random.randint(0, 2147483647))
        try:
            result = 0
            fo = self._get_GIO(tmppath)
            fi = self._get_GIO(relpath)
            fout = fo.create()
            try:
                info = GioStatResult(fi)
                result = info.st_size
                fin = fi.read()
                self._pump(fin, fout)
                fin.close()
            except gio.Error as e:
                if e.code != gio.ERROR_NOT_FOUND:
                    self._translate_gio_error(e, relpath)
            length = self._pump(file, fout)
            fout.close()
            info = GioStatResult(fo)
            if info.st_size != result + length:
                raise errors.BzrError('Failed to append size after (%d) is not original (%d) + written (%d) total (%d)' % (info.st_size, result, length, result + length))
            fo.move(fi, flags=gio.FILE_COPY_OVERWRITE)
            return result
        except gio.Error as e:
            self._translate_gio_error(e, relpath)

    def _setmode(self, relpath, mode):
        """Set permissions on a path.

        Only set permissions on Unix systems
        """
        if 'gio' in debug.debug_flags:
            mutter('GIO _setmode %s' % relpath)
        if mode:
            try:
                f = self._get_GIO(relpath)
                f.set_attribute_uint32(gio.FILE_ATTRIBUTE_UNIX_MODE, mode)
            except gio.Error as e:
                if e.code == gio.ERROR_NOT_SUPPORTED:
                    mutter('GIO Could not set permissions to %s on %s. %s', oct(mode), self._remote_path(relpath), str(e))
                else:
                    self._translate_gio_error(e, relpath)

    def rename(self, rel_from, rel_to):
        """Rename without special overwriting"""
        try:
            if 'gio' in debug.debug_flags:
                mutter('GIO move (rename): %s => %s', rel_from, rel_to)
            f = self._get_GIO(rel_from)
            t = self._get_GIO(rel_to)
            f.move(t)
        except gio.Error as e:
            self._translate_gio_error(e, rel_from)

    def move(self, rel_from, rel_to):
        """Move the item at rel_from to the location at rel_to"""
        try:
            if 'gio' in debug.debug_flags:
                mutter('GIO move: %s => %s', rel_from, rel_to)
            f = self._get_GIO(rel_from)
            t = self._get_GIO(rel_to)
            f.move(t, flags=gio.FILE_COPY_OVERWRITE)
        except gio.Error as e:
            self._translate_gio_error(e, relfrom)

    def delete(self, relpath):
        """Delete the item at relpath"""
        try:
            if 'gio' in debug.debug_flags:
                mutter('GIO delete: %s', relpath)
            f = self._get_GIO(relpath)
            f.delete()
        except gio.Error as e:
            self._translate_gio_error(e, relpath)

    def external_url(self):
        """See breezy.transport.Transport.external_url."""
        if 'gio' in debug.debug_flags:
            mutter('GIO external_url', self.base)
        return self.base

    def listable(self):
        """See Transport.listable."""
        if 'gio' in debug.debug_flags:
            mutter('GIO listable')
        return True

    def list_dir(self, relpath):
        """See Transport.list_dir."""
        if 'gio' in debug.debug_flags:
            mutter('GIO list_dir')
        try:
            entries = []
            f = self._get_GIO(relpath)
            children = f.enumerate_children(gio.FILE_ATTRIBUTE_STANDARD_NAME)
            for child in children:
                entries.append(urlutils.escape(child.get_name()))
            return entries
        except gio.Error as e:
            self._translate_gio_error(e, relpath)

    def iter_files_recursive(self):
        """See Transport.iter_files_recursive.

        This is cargo-culted from the SFTP transport"""
        if 'gio' in debug.debug_flags:
            mutter('GIO iter_files_recursive')
        queue = list(self.list_dir('.'))
        while queue:
            relpath = queue.pop(0)
            st = self.stat(relpath)
            if stat.S_ISDIR(st.st_mode):
                for i, basename in enumerate(self.list_dir(relpath)):
                    queue.insert(i, relpath + '/' + basename)
            else:
                yield relpath

    def stat(self, relpath):
        """Return the stat information for a file."""
        try:
            if 'gio' in debug.debug_flags:
                mutter('GIO stat: %s', relpath)
            f = self._get_GIO(relpath)
            return GioStatResult(f)
        except gio.Error as e:
            self._translate_gio_error(e, relpath, extra='error w/ stat')

    def lock_read(self, relpath):
        """Lock the given file for shared (read) access.
        :return: A lock object, which should be passed to Transport.unlock()
        """
        if 'gio' in debug.debug_flags:
            mutter('GIO lock_read', relpath)

        class BogusLock:

            def __init__(self, path):
                self.path = path

            def unlock(self):
                pass
        return BogusLock(relpath)

    def lock_write(self, relpath):
        """Lock the given file for exclusive (write) access.
        WARNING: many transports do not support this, so trying avoid using it

        :return: A lock object, whichshould be passed to Transport.unlock()
        """
        if 'gio' in debug.debug_flags:
            mutter('GIO lock_write', relpath)
        return self.lock_read(relpath)

    def _translate_gio_error(self, err, path, extra=None):
        if 'gio' in debug.debug_flags:
            mutter('GIO Error: {} {}'.format(str(err), path))
        if extra is None:
            extra = str(err)
        if err.code == gio.ERROR_NOT_FOUND:
            raise NoSuchFile(path, extra=extra)
        elif err.code == gio.ERROR_EXISTS:
            raise FileExists(path, extra=extra)
        elif err.code == gio.ERROR_NOT_DIRECTORY:
            raise errors.NotADirectory(path, extra=extra)
        elif err.code == gio.ERROR_NOT_EMPTY:
            raise errors.DirectoryNotEmpty(path, extra=extra)
        elif err.code == gio.ERROR_BUSY:
            raise errors.ResourceBusy(path, extra=extra)
        elif err.code == gio.ERROR_PERMISSION_DENIED:
            raise errors.PermissionDenied(path, extra=extra)
        elif err.code == gio.ERROR_HOST_NOT_FOUND:
            raise errors.PathError(path, extra=extra)
        elif err.code == gio.ERROR_IS_DIRECTORY:
            raise errors.PathError(path, extra=extra)
        else:
            mutter('unable to understand error for path: %s: %s', path, err)
            raise errors.PathError(path, extra='Unhandled gio error: ' + str(err))