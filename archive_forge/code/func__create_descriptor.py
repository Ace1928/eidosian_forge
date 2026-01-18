from __future__ import annotations
from dataclasses import is_dataclass
import inspect
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import util as orm_util
from .base import _DeclarativeMapped
from .base import LoaderCallableStatus
from .base import Mapped
from .base import PassiveFlag
from .base import SQLORMOperations
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .util import _none_set
from .util import de_stringify_annotation
from .. import event
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import util
from ..sql import expression
from ..sql import operators
from ..sql.elements import BindParameter
from ..util.typing import is_fwd_ref
from ..util.typing import is_pep593
from ..util.typing import typing_get_args
def _create_descriptor(self) -> None:
    """Create the Python descriptor that will serve as
        the access point on instances of the mapped class.

        """

    def fget(instance: Any) -> Any:
        dict_ = attributes.instance_dict(instance)
        state = attributes.instance_state(instance)
        if self.key not in dict_:
            values = [getattr(instance, key) for key in self._attribute_keys]
            if self.key not in dict_ and (state.key is not None or not _none_set.issuperset(values)):
                dict_[self.key] = self.composite_class(*values)
                state.manager.dispatch.refresh(state, self._COMPOSITE_FGET, [self.key])
        return dict_.get(self.key, None)

    def fset(instance: Any, value: Any) -> None:
        dict_ = attributes.instance_dict(instance)
        state = attributes.instance_state(instance)
        attr = state.manager[self.key]
        if attr.dispatch._active_history:
            previous = fget(instance)
        else:
            previous = dict_.get(self.key, LoaderCallableStatus.NO_VALUE)
        for fn in attr.dispatch.set:
            value = fn(state, value, previous, attr.impl)
        dict_[self.key] = value
        if value is None:
            for key in self._attribute_keys:
                setattr(instance, key, None)
        else:
            for key, value in zip(self._attribute_keys, self._composite_values_from_instance(value)):
                setattr(instance, key, value)

    def fdel(instance: Any) -> None:
        state = attributes.instance_state(instance)
        dict_ = attributes.instance_dict(instance)
        attr = state.manager[self.key]
        if attr.dispatch._active_history:
            previous = fget(instance)
            dict_.pop(self.key, None)
        else:
            previous = dict_.pop(self.key, LoaderCallableStatus.NO_VALUE)
        attr = state.manager[self.key]
        attr.dispatch.remove(state, previous, attr.impl)
        for key in self._attribute_keys:
            setattr(instance, key, None)
    self.descriptor = property(fget, fset, fdel)