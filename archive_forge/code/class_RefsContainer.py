import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
class RefsContainer:
    """A container for refs."""

    def __init__(self, logger=None) -> None:
        self._logger = logger

    def _log(self, ref, old_sha, new_sha, committer=None, timestamp=None, timezone=None, message=None):
        if self._logger is None:
            return
        if message is None:
            return
        self._logger(ref, old_sha, new_sha, committer, timestamp, timezone, message)

    def set_symbolic_ref(self, name, other, committer=None, timestamp=None, timezone=None, message=None):
        """Make a ref point at another ref.

        Args:
          name: Name of the ref to set
          other: Name of the ref to point at
          message: Optional message
        """
        raise NotImplementedError(self.set_symbolic_ref)

    def get_packed_refs(self):
        """Get contents of the packed-refs file.

        Returns: Dictionary mapping ref names to SHA1s

        Note: Will return an empty dictionary when no packed-refs file is
            present.
        """
        raise NotImplementedError(self.get_packed_refs)

    def add_packed_refs(self, new_refs: Dict[Ref, Optional[ObjectID]]):
        """Add the given refs as packed refs.

        Args:
          new_refs: A mapping of ref names to targets; if a target is None that
            means remove the ref
        """
        raise NotImplementedError(self.add_packed_refs)

    def get_peeled(self, name):
        """Return the cached peeled value of a ref, if available.

        Args:
          name: Name of the ref to peel
        Returns: The peeled value of the ref. If the ref is known not point to
            a tag, this will be the SHA the ref refers to. If the ref may point
            to a tag, but no cached information is available, None is returned.
        """
        return None

    def import_refs(self, base: Ref, other: Dict[Ref, ObjectID], committer: Optional[bytes]=None, timestamp: Optional[bytes]=None, timezone: Optional[bytes]=None, message: Optional[bytes]=None, prune: bool=False):
        if prune:
            to_delete = set(self.subkeys(base))
        else:
            to_delete = set()
        for name, value in other.items():
            if value is None:
                to_delete.add(name)
            else:
                self.set_if_equals(b'/'.join((base, name)), None, value, message=message)
            if to_delete:
                try:
                    to_delete.remove(name)
                except KeyError:
                    pass
        for ref in to_delete:
            self.remove_if_equals(b'/'.join((base, ref)), None, message=message)

    def allkeys(self):
        """All refs present in this container."""
        raise NotImplementedError(self.allkeys)

    def __iter__(self):
        return iter(self.allkeys())

    def keys(self, base=None):
        """Refs present in this container.

        Args:
          base: An optional base to return refs under.
        Returns: An unsorted set of valid refs in this container, including
            packed refs.
        """
        if base is not None:
            return self.subkeys(base)
        else:
            return self.allkeys()

    def subkeys(self, base):
        """Refs present in this container under a base.

        Args:
          base: The base to return refs under.
        Returns: A set of valid refs in this container under the base; the base
            prefix is stripped from the ref names returned.
        """
        keys = set()
        base_len = len(base) + 1
        for refname in self.allkeys():
            if refname.startswith(base):
                keys.add(refname[base_len:])
        return keys

    def as_dict(self, base=None):
        """Return the contents of this container as a dictionary."""
        ret = {}
        keys = self.keys(base)
        if base is None:
            base = b''
        else:
            base = base.rstrip(b'/')
        for key in keys:
            try:
                ret[key] = self[(base + b'/' + key).strip(b'/')]
            except (SymrefLoop, KeyError):
                continue
        return ret

    def _check_refname(self, name):
        """Ensure a refname is valid and lives in refs or is HEAD.

        HEAD is not a valid refname according to git-check-ref-format, but this
        class needs to be able to touch HEAD. Also, check_ref_format expects
        refnames without the leading 'refs/', but this class requires that
        so it cannot touch anything outside the refs dir (or HEAD).

        Args:
          name: The name of the reference.

        Raises:
          KeyError: if a refname is not HEAD or is otherwise not valid.
        """
        if name in (HEADREF, b'refs/stash'):
            return
        if not name.startswith(b'refs/') or not check_ref_format(name[5:]):
            raise RefFormatError(name)

    def read_ref(self, refname):
        """Read a reference without following any references.

        Args:
          refname: The name of the reference
        Returns: The contents of the ref file, or None if it does
            not exist.
        """
        contents = self.read_loose_ref(refname)
        if not contents:
            contents = self.get_packed_refs().get(refname, None)
        return contents

    def read_loose_ref(self, name):
        """Read a loose reference and return its contents.

        Args:
          name: the refname to read
        Returns: The contents of the ref file, or None if it does
            not exist.
        """
        raise NotImplementedError(self.read_loose_ref)

    def follow(self, name):
        """Follow a reference name.

        Returns: a tuple of (refnames, sha), wheres refnames are the names of
            references in the chain
        """
        contents = SYMREF + name
        depth = 0
        refnames = []
        while contents.startswith(SYMREF):
            refname = contents[len(SYMREF):]
            refnames.append(refname)
            contents = self.read_ref(refname)
            if not contents:
                break
            depth += 1
            if depth > 5:
                raise SymrefLoop(name, depth)
        return (refnames, contents)

    def __contains__(self, refname) -> bool:
        if self.read_ref(refname):
            return True
        return False

    def __getitem__(self, name):
        """Get the SHA1 for a reference name.

        This method follows all symbolic references.
        """
        _, sha = self.follow(name)
        if sha is None:
            raise KeyError(name)
        return sha

    def set_if_equals(self, name, old_ref, new_ref, committer=None, timestamp=None, timezone=None, message=None):
        """Set a refname to new_ref only if it currently equals old_ref.

        This method follows all symbolic references if applicable for the
        subclass, and can be used to perform an atomic compare-and-swap
        operation.

        Args:
          name: The refname to set.
          old_ref: The old sha the refname must refer to, or None to set
            unconditionally.
          new_ref: The new sha the refname will refer to.
          message: Message for reflog
        Returns: True if the set was successful, False otherwise.
        """
        raise NotImplementedError(self.set_if_equals)

    def add_if_new(self, name, ref, committer=None, timestamp=None, timezone=None, message=None):
        """Add a new reference only if it does not already exist.

        Args:
          name: Ref name
          ref: Ref value
        """
        raise NotImplementedError(self.add_if_new)

    def __setitem__(self, name, ref) -> None:
        """Set a reference name to point to the given SHA1.

        This method follows all symbolic references if applicable for the
        subclass.

        Note: This method unconditionally overwrites the contents of a
            reference. To update atomically only if the reference has not
            changed, use set_if_equals().

        Args:
          name: The refname to set.
          ref: The new sha the refname will refer to.
        """
        self.set_if_equals(name, None, ref)

    def remove_if_equals(self, name, old_ref, committer=None, timestamp=None, timezone=None, message=None):
        """Remove a refname only if it currently equals old_ref.

        This method does not follow symbolic references, even if applicable for
        the subclass. It can be used to perform an atomic compare-and-delete
        operation.

        Args:
          name: The refname to delete.
          old_ref: The old sha the refname must refer to, or None to
            delete unconditionally.
          message: Message for reflog
        Returns: True if the delete was successful, False otherwise.
        """
        raise NotImplementedError(self.remove_if_equals)

    def __delitem__(self, name) -> None:
        """Remove a refname.

        This method does not follow symbolic references, even if applicable for
        the subclass.

        Note: This method unconditionally deletes the contents of a reference.
            To delete atomically only if the reference has not changed, use
            remove_if_equals().

        Args:
          name: The refname to delete.
        """
        self.remove_if_equals(name, None)

    def get_symrefs(self):
        """Get a dict with all symrefs in this container.

        Returns: Dictionary mapping source ref to target ref
        """
        ret = {}
        for src in self.allkeys():
            try:
                dst = parse_symref_value(self.read_ref(src))
            except ValueError:
                pass
            else:
                ret[src] = dst
        return ret