import os
from typing import Set
from pyinotify import (IN_ATTRIB, IN_CLOSE_WRITE, IN_CREATE, IN_DELETE,
from .workingtree import WorkingTree
class DirtyTracker:
    """Track the changes to (part of) a working tree."""
    _process: _Process

    def __init__(self, tree: WorkingTree, subpath: str='.') -> None:
        self._tree = tree
        self._subpath = subpath

    def __enter__(self):
        try:
            self._wm = WatchManager()
        except OSError as e:
            if 'EMFILE' in e.args[0]:
                raise TooManyOpenFiles()
            raise
        self._process = _Process()
        self._notifier = Notifier(self._wm, self._process)
        self._notifier.coalesce_events(True)

        def check_excluded(p: str) -> bool:
            return self._tree.is_control_filename(self._tree.relpath(p))
        self._wdd = self._wm.add_watch(self._tree.abspath(self._subpath), MASK, rec=True, auto_add=True, exclude_filter=check_excluded)
        return self

    def __exit__(self, exc_val, exc_typ, exc_tb):
        self._wdd.clear()
        self._wm.close()
        return False

    def _process_pending(self) -> None:
        if self._notifier.check_events(timeout=0):
            self._notifier.read_events()
        self._notifier.process_events()

    def mark_clean(self) -> None:
        """Mark the subtree as not having any changes."""
        self._process_pending()
        self._process.paths.clear()
        self._process.created.clear()

    def is_dirty(self) -> bool:
        """Check whether there are any changes."""
        self._process_pending()
        return bool(self._paths)

    def paths(self) -> Set[str]:
        """Return the paths that have changed."""
        self._process_pending()
        return self._paths

    @property
    def _paths(self) -> Set[str]:
        return self._process.paths

    @property
    def _created(self):
        return self._process.created

    def relpaths(self) -> Set[str]:
        """Return the paths relative to the tree root that changed."""
        return {self._tree.relpath(p) for p in self.paths()}