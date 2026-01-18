import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
class DiskRefsContainer(RefsContainer):
    """Refs container that reads refs from disk."""

    def __init__(self, path, worktree_path=None, logger=None) -> None:
        super().__init__(logger=logger)
        if getattr(path, 'encode', None) is not None:
            path = os.fsencode(path)
        self.path = path
        if worktree_path is None:
            worktree_path = path
        if getattr(worktree_path, 'encode', None) is not None:
            worktree_path = os.fsencode(worktree_path)
        self.worktree_path = worktree_path
        self._packed_refs = None
        self._peeled_refs = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.path!r})'

    def subkeys(self, base):
        subkeys = set()
        path = self.refpath(base)
        for root, unused_dirs, files in os.walk(path):
            dir = root[len(path):]
            if os.path.sep != '/':
                dir = dir.replace(os.fsencode(os.path.sep), b'/')
            dir = dir.strip(b'/')
            for filename in files:
                refname = b'/'.join(([dir] if dir else []) + [filename])
                if check_ref_format(base + b'/' + refname):
                    subkeys.add(refname)
        for key in self.get_packed_refs():
            if key.startswith(base):
                subkeys.add(key[len(base):].strip(b'/'))
        return subkeys

    def allkeys(self):
        allkeys = set()
        if os.path.exists(self.refpath(HEADREF)):
            allkeys.add(HEADREF)
        path = self.refpath(b'')
        refspath = self.refpath(b'refs')
        for root, unused_dirs, files in os.walk(refspath):
            dir = root[len(path):]
            if os.path.sep != '/':
                dir = dir.replace(os.fsencode(os.path.sep), b'/')
            for filename in files:
                refname = b'/'.join([dir, filename])
                if check_ref_format(refname):
                    allkeys.add(refname)
        allkeys.update(self.get_packed_refs())
        return allkeys

    def refpath(self, name):
        """Return the disk path of a ref."""
        if os.path.sep != '/':
            name = name.replace(b'/', os.fsencode(os.path.sep))
        if name == HEADREF:
            return os.path.join(self.worktree_path, name)
        else:
            return os.path.join(self.path, name)

    def get_packed_refs(self):
        """Get contents of the packed-refs file.

        Returns: Dictionary mapping ref names to SHA1s

        Note: Will return an empty dictionary when no packed-refs file is
            present.
        """
        if self._packed_refs is None:
            self._packed_refs = {}
            self._peeled_refs = {}
            path = os.path.join(self.path, b'packed-refs')
            try:
                f = GitFile(path, 'rb')
            except FileNotFoundError:
                return {}
            with f:
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
        return self._packed_refs

    def add_packed_refs(self, new_refs: Dict[Ref, Optional[ObjectID]]):
        """Add the given refs as packed refs.

        Args:
          new_refs: A mapping of ref names to targets; if a target is None that
            means remove the ref
        """
        if not new_refs:
            return
        path = os.path.join(self.path, b'packed-refs')
        with GitFile(path, 'wb') as f:
            packed_refs = self.get_packed_refs().copy()
            for ref, target in new_refs.items():
                if ref == HEADREF:
                    raise ValueError('cannot pack HEAD')
                with suppress(OSError):
                    os.remove(self.refpath(ref))
                if target is not None:
                    packed_refs[ref] = target
                else:
                    packed_refs.pop(ref, None)
            write_packed_refs(f, packed_refs, self._peeled_refs)
            self._packed_refs = packed_refs

    def get_peeled(self, name):
        """Return the cached peeled value of a ref, if available.

        Args:
          name: Name of the ref to peel
        Returns: The peeled value of the ref. If the ref is known not point to
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

        Args:
          name: the refname to read, relative to refpath
        Returns: The contents of the ref file, or None if the file does not
            exist.

        Raises:
          IOError: if any other error occurs
        """
        filename = self.refpath(name)
        try:
            with GitFile(filename, 'rb') as f:
                header = f.read(len(SYMREF))
                if header == SYMREF:
                    return header + next(iter(f)).rstrip(b'\r\n')
                else:
                    return header + f.read(40 - len(SYMREF))
        except (OSError, UnicodeError):
            return None

    def _remove_packed_ref(self, name):
        if self._packed_refs is None:
            return
        filename = os.path.join(self.path, b'packed-refs')
        f = GitFile(filename, 'wb')
        try:
            self._packed_refs = None
            self.get_packed_refs()
            if name not in self._packed_refs:
                return
            del self._packed_refs[name]
            with suppress(KeyError):
                del self._peeled_refs[name]
            write_packed_refs(f, self._packed_refs, self._peeled_refs)
            f.close()
        finally:
            f.abort()

    def set_symbolic_ref(self, name, other, committer=None, timestamp=None, timezone=None, message=None):
        """Make a ref point at another ref.

        Args:
          name: Name of the ref to set
          other: Name of the ref to point at
          message: Optional message to describe the change
        """
        self._check_refname(name)
        self._check_refname(other)
        filename = self.refpath(name)
        f = GitFile(filename, 'wb')
        try:
            f.write(SYMREF + other + b'\n')
            sha = self.follow(name)[-1]
            self._log(name, sha, sha, committer=committer, timestamp=timestamp, timezone=timezone, message=message)
        except BaseException:
            f.abort()
            raise
        else:
            f.close()

    def set_if_equals(self, name, old_ref, new_ref, committer=None, timestamp=None, timezone=None, message=None):
        """Set a refname to new_ref only if it currently equals old_ref.

        This method follows all symbolic references, and can be used to perform
        an atomic compare-and-swap operation.

        Args:
          name: The refname to set.
          old_ref: The old sha the refname must refer to, or None to set
            unconditionally.
          new_ref: The new sha the refname will refer to.
          message: Set message for reflog
        Returns: True if the set was successful, False otherwise.
        """
        self._check_refname(name)
        try:
            realnames, _ = self.follow(name)
            realname = realnames[-1]
        except (KeyError, IndexError, SymrefLoop):
            realname = name
        filename = self.refpath(realname)
        probe_ref = os.path.dirname(realname)
        packed_refs = self.get_packed_refs()
        while probe_ref:
            if packed_refs.get(probe_ref, None) is not None:
                raise NotADirectoryError(filename)
            probe_ref = os.path.dirname(probe_ref)
        ensure_dir_exists(os.path.dirname(filename))
        with GitFile(filename, 'wb') as f:
            if old_ref is not None:
                try:
                    orig_ref = self.read_loose_ref(realname)
                    if orig_ref is None:
                        orig_ref = self.get_packed_refs().get(realname, ZERO_SHA)
                    if orig_ref != old_ref:
                        f.abort()
                        return False
                except OSError:
                    f.abort()
                    raise
            try:
                f.write(new_ref + b'\n')
            except OSError:
                f.abort()
                raise
            self._log(realname, old_ref, new_ref, committer=committer, timestamp=timestamp, timezone=timezone, message=message)
        return True

    def add_if_new(self, name: bytes, ref: bytes, committer=None, timestamp=None, timezone=None, message: Optional[bytes]=None):
        """Add a new reference only if it does not already exist.

        This method follows symrefs, and only ensures that the last ref in the
        chain does not exist.

        Args:
          name: The refname to set.
          ref: The new sha the refname will refer to.
          message: Optional message for reflog
        Returns: True if the add was successful, False otherwise.
        """
        try:
            realnames, contents = self.follow(name)
            if contents is not None:
                return False
            realname = realnames[-1]
        except (KeyError, IndexError):
            realname = name
        self._check_refname(realname)
        filename = self.refpath(realname)
        ensure_dir_exists(os.path.dirname(filename))
        with GitFile(filename, 'wb') as f:
            if os.path.exists(filename) or name in self.get_packed_refs():
                f.abort()
                return False
            try:
                f.write(ref + b'\n')
            except OSError:
                f.abort()
                raise
            else:
                self._log(name, None, ref, committer=committer, timestamp=timestamp, timezone=timezone, message=message)
        return True

    def remove_if_equals(self, name, old_ref, committer=None, timestamp=None, timezone=None, message=None):
        """Remove a refname only if it currently equals old_ref.

        This method does not follow symbolic references. It can be used to
        perform an atomic compare-and-delete operation.

        Args:
          name: The refname to delete.
          old_ref: The old sha the refname must refer to, or None to
            delete unconditionally.
          message: Optional message
        Returns: True if the delete was successful, False otherwise.
        """
        self._check_refname(name)
        filename = self.refpath(name)
        ensure_dir_exists(os.path.dirname(filename))
        f = GitFile(filename, 'wb')
        try:
            if old_ref is not None:
                orig_ref = self.read_loose_ref(name)
                if orig_ref is None:
                    orig_ref = self.get_packed_refs().get(name, ZERO_SHA)
                if orig_ref != old_ref:
                    return False
            try:
                found = os.path.lexists(filename)
            except OSError:
                found = False
            if found:
                os.remove(filename)
            self._remove_packed_ref(name)
            self._log(name, old_ref, None, committer=committer, timestamp=timestamp, timezone=timezone, message=message)
        finally:
            f.abort()
        parent = name
        while True:
            try:
                parent, _ = parent.rsplit(b'/', 1)
            except ValueError:
                break
            if parent == b'refs':
                break
            parent_filename = self.refpath(parent)
            try:
                os.rmdir(parent_filename)
            except OSError:
                break
        return True