from __future__ import annotations
from typing import Any
from streamlit import util
from streamlit.runtime.scriptrunner import get_script_run_ctx
class LockedCursor(Cursor):

    def __init__(self, root_container: int, parent_path: tuple[int, ...]=(), index: int=0, **props):
        """A locked pointer to a location in the app.

        LockedCursors always point to the same location, even when you call
        get_locked_cursor() on them.

        Parameters
        ----------
        root_container: int
            The root container this cursor lives in.
        parent_path: tuple of ints
          The full path of this cursor, consisting of the IDs of all ancestors. The
          0th item is the topmost ancestor.
        index: int
        **props: any
          Anything else you want to store in this cursor. This is a temporary
          measure that will go away when we implement improved return values
          for elements.

        """
        self._root_container = root_container
        self._index = index
        self._parent_path = parent_path
        self._props = props

    @property
    def root_container(self) -> int:
        return self._root_container

    @property
    def parent_path(self) -> tuple[int, ...]:
        return self._parent_path

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_locked(self) -> bool:
        return True

    def get_locked_cursor(self, **props) -> LockedCursor:
        self._props = props
        return self

    @property
    def props(self) -> Any:
        return self._props