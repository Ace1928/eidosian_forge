import copy
import threading
import contextlib
import operator
import copyreg
from typing import Any, Type, Tuple, Dict, List, Union, Optional, Callable, TypeVar, Iterable, Generic, TYPE_CHECKING
def _setup_init(self):
    """
        Setup and initialize the proxy object arguments
        """
    from lazyops.utils.helpers import lazy_import
    if self.__dict__['__obj_args_'] is not None and (not isinstance(self.__dict__['__obj_args_'], (list, tuple))):
        if isinstance(self.__dict__['__obj_args_'], str):
            self.__dict__['__obj_args_'] = lazy_import(self.__dict__['__obj_args_'])
        if callable(self.__dict__['__obj_args_']):
            self.__dict__['__obj_args_'] = self.__dict__['__obj_args_']()
    if self.__dict__['__obj_kwargs_'] is not None and (not isinstance(self.__dict__['__obj_kwargs_'], dict)):
        if isinstance(self.__dict__['__obj_kwargs_'], str):
            self.__dict__['__obj_kwargs_'] = lazy_import(self.__dict__['__obj_kwargs_'])
        if callable(self.__dict__['__obj_kwargs_']):
            self.__dict__['__obj_kwargs_'] = self.__dict__['__obj_kwargs_']()
    if self.__dict__['__obj_getter_'] is not None and isinstance(self.__dict__['__obj_getter_'], str):
        self.__dict__['__obj_getter_'] = lazy_import(self.__dict__['__obj_getter_'])
    elif self.__dict__['__obj_cls_'] is not None and isinstance(self.__dict__['__obj_cls_'], str):
        self.__dict__['__obj_cls_'] = lazy_import(self.__dict__['__obj_cls_'])