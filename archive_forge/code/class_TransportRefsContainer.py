import os
import posixpath
import sys
from io import BytesIO
from dulwich.errors import NoIndexPresent
from dulwich.file import FileLocked, _GitFile
from dulwich.object_store import (PACK_MODE, PACKDIR, PackBasedObjectStore,
from dulwich.objects import ShaFile
from dulwich.pack import (PACK_SPOOL_FILE_MAX_SIZE, MemoryPackIndex, Pack,
from dulwich.refs import SymrefLoop
from dulwich.repo import (BASE_DIRECTORIES, COMMONDIR, CONTROLDIR,
from .. import osutils
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..errors import (AlreadyControlDirError, LockBroken, LockContention,
from ..lock import LogicalLockResult
from ..trace import warning
from ..transport import FileExists, NoSuchFile
from ..transport.local import LocalTransport
class TransportRefsContainer(RefsContainer):
    """Refs container that reads refs from a transport."""

    def __init__(self, transport, worktree_transport=None):
        self.transport = transport
        if worktree_transport is None:
            worktree_transport = transport
        self.worktree_transport = worktree_transport
        self._packed_refs = None
        self._peeled_refs = None

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.transport)

    def _ensure_dir_exists(self, path):
        self.transport.clone(posixpath.dirname(path)).create_prefix()

    def subkeys(self, base):
        """Refs present in this container under a base.

        :param base: The base to return refs under.
        :return: A set of valid refs in this container under the base; the base
            prefix is stripped from the ref names returned.
        """
        keys = set()
        base_len = len(base) + 1
        for refname in self.allkeys():
            if refname.startswith(base):
                keys.add(refname[base_len:])
        return keys

    def allkeys(self):
        keys = set()
        try:
            self.worktree_transport.get_bytes('HEAD')
        except NoSuchFile:
            pass
        else:
            keys.add(b'HEAD')
        try:
            iter_files = list(self.transport.clone('refs').iter_files_recursive())
            for filename in iter_files:
                unquoted_filename = urlutils.unquote_to_bytes(filename)
                refname = osutils.pathjoin(b'refs', unquoted_filename)
                if check_ref_format(refname):
                    keys.add(refname)
        except (TransportNotPossible, NoSuchFile):
            pass
        keys.update(self.get_packed_refs())
        return keys

    def get_packed_refs(self):
        """Get contents of the packed-refs file.

        :return: Dictionary mapping ref names to SHA1s

        :note: Will return an empty dictionary when no packed-refs file is
            present.
        """
        if self._packed_refs is None:
            self._packed_refs = {}
            self._peeled_refs = {}
            try:
                f = self.transport.get('packed-refs')
            except NoSuchFile:
                return {}
            try:
                first_line = next(iter(f)).rstrip()
                if first_line.startswith(b'# pack-refs') and b' peeled' in first_line:
                    for sha, name, peeled in read_packed_refs_with_peeled(f):
                        self._packed_refs[name] = sha
                        if peeled:
                            self._peeled_refs[name] = peeled
                else:
                    f.seek(0)
                    for sha, name in read_packed_refs(f):
                        self._packed_refs[name] = sha
            finally:
                f.close()
        return self._packed_refs

    def get_peeled(self, name):
        """Return the cached peeled value of a ref, if available.

        :param name: Name of the ref to peel
        :return: The peeled value of the ref. If the ref is known not point to
            a tag, this will be the SHA the ref refers to. If the ref may point
            to a tag, but no cached information is available, None is returned.
        """
        self.get_packed_refs()
        if self._peeled_refs is None or name not in self._packed_refs:
            return None
        if name in self._peeled_refs:
            return self._peeled_refs[name]
        else:
            return self[name]

    def read_loose_ref(self, name):
        """Read a reference file and return its contents.

        If the reference file a symbolic reference, only read the first line of
        the file. Otherwise, only read the first 40 bytes.

        :param name: the refname to read, relative to refpath
        :return: The contents of the ref file, or None if the file does not
            exist.
        :raises IOError: if any other error occurs
        """
        if name == b'HEAD':
            transport = self.worktree_transport
        else:
            transport = self.transport
        try:
            f = transport.get(urlutils.quote_from_bytes(name))
        except NoSuchFile:
            return None
        with f:
            try:
                header = f.read(len(SYMREF))
            except ReadError:
                return None
            if header == SYMREF:
                return header + next(iter(f)).rstrip(b'\r\n')
            else:
                return header + f.read(40 - len(SYMREF))

    def _remove_packed_ref(self, name):
        if self._packed_refs is None:
            return
        self._packed_refs = None
        self.get_packed_refs()
        if name not in self._packed_refs:
            return
        del self._packed_refs[name]
        if name in self._peeled_refs:
            del self._peeled_refs[name]
        with self.transport.open_write_stream('packed-refs') as f:
            write_packed_refs(f, self._packed_refs, self._peeled_refs)

    def set_symbolic_ref(self, name, other):
        """Make a ref point at another ref.

        :param name: Name of the ref to set
        :param other: Name of the ref to point at
        """
        self._check_refname(name)
        self._check_refname(other)
        if name != b'HEAD':
            transport = self.transport
            self._ensure_dir_exists(urlutils.quote_from_bytes(name))
        else:
            transport = self.worktree_transport
        transport.put_bytes(urlutils.quote_from_bytes(name), SYMREF + other + b'\n')

    def set_if_equals(self, name, old_ref, new_ref):
        """Set a refname to new_ref only if it currently equals old_ref.

        This method follows all symbolic references, and can be used to perform
        an atomic compare-and-swap operation.

        :param name: The refname to set.
        :param old_ref: The old sha the refname must refer to, or None to set
            unconditionally.
        :param new_ref: The new sha the refname will refer to.
        :return: True if the set was successful, False otherwise.
        """
        self._check_refname(name)
        try:
            realnames, _ = self.follow(name)
            realname = realnames[-1]
        except (KeyError, IndexError, SymrefLoop):
            realname = name
        if realname == b'HEAD':
            transport = self.worktree_transport
        else:
            transport = self.transport
            self._ensure_dir_exists(urlutils.quote_from_bytes(realname))
        transport.put_bytes(urlutils.quote_from_bytes(realname), new_ref + b'\n')
        return True

    def add_if_new(self, name, ref):
        """Add a new reference only if it does not already exist.

        This method follows symrefs, and only ensures that the last ref in the
        chain does not exist.

        :param name: The refname to set.
        :param ref: The new sha the refname will refer to.
        :return: True if the add was successful, False otherwise.
        """
        try:
            realnames, contents = self.follow(name)
            if contents is not None:
                return False
            realname = realnames[-1]
        except (KeyError, IndexError):
            realname = name
        self._check_refname(realname)
        if realname == b'HEAD':
            transport = self.worktree_transport
        else:
            transport = self.transport
            self._ensure_dir_exists(urlutils.quote_from_bytes(realname))
        transport.put_bytes(urlutils.quote_from_bytes(realname), ref + b'\n')
        return True

    def remove_if_equals(self, name, old_ref):
        """Remove a refname only if it currently equals old_ref.

        This method does not follow symbolic references. It can be used to
        perform an atomic compare-and-delete operation.

        :param name: The refname to delete.
        :param old_ref: The old sha the refname must refer to, or None to
            delete unconditionally.
        :return: True if the delete was successful, False otherwise.
        """
        self._check_refname(name)
        if name == b'HEAD':
            transport = self.worktree_transport
        else:
            transport = self.transport
        try:
            transport.delete(urlutils.quote_from_bytes(name))
        except NoSuchFile:
            pass
        self._remove_packed_ref(name)
        return True

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def unlock_ref(self, name):
        if name == b'HEAD':
            transport = self.worktree_transport
        else:
            transport = self.transport
        lockname = name + b'.lock'
        try:
            transport.delete(urlutils.quote_from_bytes(lockname))
        except NoSuchFile:
            pass

    def lock_ref(self, name):
        if name == b'HEAD':
            transport = self.worktree_transport
        else:
            transport = self.transport
        self._ensure_dir_exists(urlutils.quote_from_bytes(name))
        lockname = urlutils.quote_from_bytes(name + b'.lock')
        try:
            local_path = transport.local_abspath(urlutils.quote_from_bytes(name))
        except NotLocalUrl:
            if transport.has(lockname):
                raise LockContention(name)
            transport.put_bytes(lockname, b'Locked by brz-git')
            return LogicalLockResult(lambda: transport.delete(lockname))
        else:
            try:
                gf = TransportGitFile(transport, urlutils.quote_from_bytes(name), 'wb')
            except FileLocked as e:
                raise LockContention(name, e)
            else:

                def unlock():
                    try:
                        transport.delete(lockname)
                    except NoSuchFile:
                        raise LockBroken(lockname)
                    gf.abort()
                return LogicalLockResult(unlock)