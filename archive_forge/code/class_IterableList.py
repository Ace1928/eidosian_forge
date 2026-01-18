from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
class IterableList(List[T_IterableObj]):
    """
    List of iterable objects allowing to query an object by id or by named index::

     heads = repo.heads
     heads.master
     heads['master']
     heads[0]

    Iterable parent objects = [Commit, SubModule, Reference, FetchInfo, PushInfo]
    Iterable via inheritance = [Head, TagReference, RemoteReference]

    It requires an id_attribute name to be set which will be queried from its
    contained items to have a means for comparison.

    A prefix can be specified which is to be used in case the id returned by the
    items always contains a prefix that does not matter to the user, so it
    can be left out.
    """
    __slots__ = ('_id_attr', '_prefix')

    def __new__(cls, id_attr: str, prefix: str='') -> 'IterableList[T_IterableObj]':
        return super().__new__(cls)

    def __init__(self, id_attr: str, prefix: str='') -> None:
        self._id_attr = id_attr
        self._prefix = prefix

    def __contains__(self, attr: object) -> bool:
        try:
            rval = list.__contains__(self, attr)
            if rval:
                return rval
        except (AttributeError, TypeError):
            pass
        try:
            getattr(self, cast(str, attr))
            return True
        except (AttributeError, TypeError):
            return False

    def __getattr__(self, attr: str) -> T_IterableObj:
        attr = self._prefix + attr
        for item in self:
            if getattr(item, self._id_attr) == attr:
                return item
        return list.__getattribute__(self, attr)

    def __getitem__(self, index: Union[SupportsIndex, int, slice, str]) -> T_IterableObj:
        assert isinstance(index, (int, str, slice)), 'Index of IterableList should be an int or str'
        if isinstance(index, int):
            return list.__getitem__(self, index)
        elif isinstance(index, slice):
            raise ValueError('Index should be an int or str')
        else:
            try:
                return getattr(self, index)
            except AttributeError as e:
                raise IndexError('No item found with id %r' % (self._prefix + index)) from e

    def __delitem__(self, index: Union[SupportsIndex, int, slice, str]) -> None:
        assert isinstance(index, (int, str)), 'Index of IterableList should be an int or str'
        delindex = cast(int, index)
        if not isinstance(index, int):
            delindex = -1
            name = self._prefix + index
            for i, item in enumerate(self):
                if getattr(item, self._id_attr) == name:
                    delindex = i
                    break
            if delindex == -1:
                raise IndexError('Item with name %s not found' % name)
        list.__delitem__(self, delindex)