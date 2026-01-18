import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
class ClassSelector(SelectorBase):
    """
    Parameter allowing selection of either a subclass or an instance of a given set of classes.
    By default, requires an instance, but if is_instance=False, accepts a class instead.
    Both class and instance values respect the instantiate slot, though it matters only
    for is_instance=True.
    """
    __slots__ = ['class_', 'is_instance']
    _slot_defaults = _dict_update(SelectorBase._slot_defaults, instantiate=True, is_instance=True)

    @typing.overload
    def __init__(self, *, class_, default=None, instantiate=True, is_instance=True, allow_None=False, doc=None, label=None, precedence=None, constant=False, readonly=False, pickle_default_value=True, per_instance=True, allow_refs=False, nested_refs=False):
        ...

    @_deprecate_positional_args
    def __init__(self, *, class_, default=Undefined, instantiate=Undefined, is_instance=Undefined, **params):
        self.class_ = class_
        self.is_instance = is_instance
        super().__init__(default=default, instantiate=instantiate, **params)
        self._validate(self.default)

    def _validate(self, val):
        super()._validate(val)
        self._validate_class_(val, self.class_, self.is_instance)

    def _validate_class_(self, val, class_, is_instance):
        if val is None and self.allow_None:
            return
        if isinstance(class_, tuple):
            class_name = '(%s)' % ', '.join((cl.__name__ for cl in class_))
        else:
            class_name = class_.__name__
        if is_instance:
            if not isinstance(val, class_):
                raise ValueError(f'{_validate_error_prefix(self)} value must be an instance of {class_name}, not {val!r}.')
        elif not issubclass(val, class_):
            raise ValueError(f'{_validate_error_prefix(self)} value must be a subclass of {class_name}, not {val}.')

    def get_range(self):
        """
        Return the possible types for this parameter's value.

        (I.e. return `{name: <class>}` for all classes that are
        concrete_descendents() of `self.class_`.)

        Only classes from modules that have been imported are added
        (see concrete_descendents()).
        """
        classes = self.class_ if isinstance(self.class_, tuple) else (self.class_,)
        all_classes = {}
        for cls in classes:
            all_classes.update(concrete_descendents(cls))
        d = OrderedDict(((name, class_) for name, class_ in all_classes.items()))
        if self.allow_None:
            d['None'] = None
        return d