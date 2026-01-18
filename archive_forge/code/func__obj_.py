import copy
import threading
import contextlib
import operator
import copyreg
from typing import Any, Type, Tuple, Dict, List, Union, Optional, Callable, TypeVar, Iterable, Generic, TYPE_CHECKING
@property
def _obj_(self) -> ProxyObjT:
    """
        Returns the object
        """
    if self.__dict__['__obj_'] is None:
        with self._objlock_():
            if self.__dict__['__obj_getter_']:
                self.__dict__['__obj_'] = self.__dict__['__obj_getter_'](*self.__dict__['__obj_args_'], **self.__dict__['__obj_kwargs_'])
            elif self.__dict__['__obj_cls_']:
                if isinstance(self.__dict__['__obj_cls_'], str):
                    from lazyops.utils.helpers import lazy_import
                    self.__dict__['__obj_cls_'] = lazy_import(self.__dict__['__obj_cls_'])
                if self.__dict__['__obj_initialize_']:
                    self.__dict__['__obj_'] = self.__dict__['__obj_cls_'](*self.__dict__['__obj_args_'], **self.__dict__['__obj_kwargs_'])
                else:
                    self.__dict__['__obj_'] = self.__dict__['__obj_cls_']
    return self.__dict__['__obj_']