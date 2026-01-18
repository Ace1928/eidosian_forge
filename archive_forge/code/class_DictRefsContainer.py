import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
class DictRefsContainer(RefsContainer):
    """RefsContainer backed by a simple dict.

    This container does not support symbolic or packed references and is not
    threadsafe.
    """

    def __init__(self, refs, logger=None) -> None:
        super().__init__(logger=logger)
        self._refs = refs
        self._peeled: Dict[bytes, ObjectID] = {}
        self._watchers: Set[Any] = set()

    def allkeys(self):
        return self._refs.keys()

    def read_loose_ref(self, name):
        return self._refs.get(name, None)

    def get_packed_refs(self):
        return {}

    def _notify(self, ref, newsha):
        for watcher in self._watchers:
            watcher._notify((ref, newsha))

    def set_symbolic_ref(self, name: Ref, other: Ref, committer=None, timestamp=None, timezone=None, message=None):
        old = self.follow(name)[-1]
        new = SYMREF + other
        self._refs[name] = new
        self._notify(name, new)
        self._log(name, old, new, committer=committer, timestamp=timestamp, timezone=timezone, message=message)

    def set_if_equals(self, name, old_ref, new_ref, committer=None, timestamp=None, timezone=None, message=None):
        if old_ref is not None and self._refs.get(name, ZERO_SHA) != old_ref:
            return False
        realnames, _ = self.follow(name)
        for realname in realnames:
            self._check_refname(realname)
            old = self._refs.get(realname)
            self._refs[realname] = new_ref
            self._notify(realname, new_ref)
            self._log(realname, old, new_ref, committer=committer, timestamp=timestamp, timezone=timezone, message=message)
        return True

    def add_if_new(self, name: Ref, ref: ObjectID, committer=None, timestamp=None, timezone=None, message: Optional[bytes]=None):
        if name in self._refs:
            return False
        self._refs[name] = ref
        self._notify(name, ref)
        self._log(name, None, ref, committer=committer, timestamp=timestamp, timezone=timezone, message=message)
        return True

    def remove_if_equals(self, name, old_ref, committer=None, timestamp=None, timezone=None, message=None):
        if old_ref is not None and self._refs.get(name, ZERO_SHA) != old_ref:
            return False
        try:
            old = self._refs.pop(name)
        except KeyError:
            pass
        else:
            self._notify(name, None)
            self._log(name, old, None, committer=committer, timestamp=timestamp, timezone=timezone, message=message)
        return True

    def get_peeled(self, name):
        return self._peeled.get(name)

    def _update(self, refs):
        """Update multiple refs; intended only for testing."""
        for ref, sha in refs.items():
            self.set_if_equals(ref, None, sha)

    def _update_peeled(self, peeled):
        """Update cached peeled refs; intended only for testing."""
        self._peeled.update(peeled)