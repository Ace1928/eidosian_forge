import copy
import importlib
import json
import pprint
import textwrap
from types import ModuleType
from typing import Any, Dict, Iterable, Optional, Type, TYPE_CHECKING, Union
import gitlab
from gitlab import types as g_types
from gitlab.exceptions import GitlabParsingError
from .client import Gitlab, GitlabList
class RESTObjectList:
    """Generator object representing a list of RESTObject's.

    This generator uses the Gitlab pagination system to fetch new data when
    required.

    Note: you should not instantiate such objects, they are returned by calls
    to RESTManager.list()

    Args:
        manager: Manager to attach to the created objects
        obj_cls: Type of objects to create from the json data
        _list: A GitlabList object
    """

    def __init__(self, manager: 'RESTManager', obj_cls: Type[RESTObject], _list: GitlabList) -> None:
        """Creates an objects list from a GitlabList.

        You should not create objects of this type, but use managers list()
        methods instead.

        Args:
            manager: the RESTManager to attach to the objects
            obj_cls: the class of the created objects
            _list: the GitlabList holding the data
        """
        self.manager = manager
        self._obj_cls = obj_cls
        self._list = _list

    def __iter__(self) -> 'RESTObjectList':
        return self

    def __len__(self) -> int:
        return len(self._list)

    def __next__(self) -> RESTObject:
        return self.next()

    def next(self) -> RESTObject:
        data = self._list.next()
        return self._obj_cls(self.manager, data, created_from_list=True)

    @property
    def current_page(self) -> int:
        """The current page number."""
        return self._list.current_page

    @property
    def prev_page(self) -> Optional[int]:
        """The previous page number.

        If None, the current page is the first.
        """
        return self._list.prev_page

    @property
    def next_page(self) -> Optional[int]:
        """The next page number.

        If None, the current page is the last.
        """
        return self._list.next_page

    @property
    def per_page(self) -> Optional[int]:
        """The number of items per page."""
        return self._list.per_page

    @property
    def total_pages(self) -> Optional[int]:
        """The total number of pages."""
        return self._list.total_pages

    @property
    def total(self) -> Optional[int]:
        """The total number of items."""
        return self._list.total