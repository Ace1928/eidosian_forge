from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import base
from .collections import collection
from .collections import collection_adapter
from .. import exc as sa_exc
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..util.typing import Literal
class KeyFuncDict(Dict[_KT, _VT]):
    """Base for ORM mapped dictionary classes.

    Extends the ``dict`` type with additional methods needed by SQLAlchemy ORM
    collection classes. Use of :class:`_orm.KeyFuncDict` is most directly
    by using the :func:`.attribute_keyed_dict` or
    :func:`.column_keyed_dict` class factories.
    :class:`_orm.KeyFuncDict` may also serve as the base for user-defined
    custom dictionary classes.

    .. versionchanged:: 2.0 Renamed :class:`.MappedCollection` to
       :class:`.KeyFuncDict`.

    .. seealso::

        :func:`_orm.attribute_keyed_dict`

        :func:`_orm.column_keyed_dict`

        :ref:`orm_dictionary_collection`

        :ref:`orm_custom_collection`


    """

    def __init__(self, keyfunc: _F, *dict_args: Any, ignore_unpopulated_attribute: bool=False) -> None:
        """Create a new collection with keying provided by keyfunc.

        keyfunc may be any callable that takes an object and returns an object
        for use as a dictionary key.

        The keyfunc will be called every time the ORM needs to add a member by
        value-only (such as when loading instances from the database) or
        remove a member.  The usual cautions about dictionary keying apply-
        ``keyfunc(object)`` should return the same output for the life of the
        collection.  Keying based on mutable properties can result in
        unreachable instances "lost" in the collection.

        """
        self.keyfunc = keyfunc
        self.ignore_unpopulated_attribute = ignore_unpopulated_attribute
        super().__init__(*dict_args)

    @classmethod
    def _unreduce(cls, keyfunc: _F, values: Dict[_KT, _KT], adapter: Optional[CollectionAdapter]=None) -> 'KeyFuncDict[_KT, _KT]':
        mp: KeyFuncDict[_KT, _KT] = KeyFuncDict(keyfunc)
        mp.update(values)
        return mp

    def __reduce__(self) -> Tuple[Callable[[_KT, _KT], KeyFuncDict[_KT, _KT]], Tuple[Any, Union[Dict[_KT, _KT], Dict[_KT, _KT]], CollectionAdapter]]:
        return (KeyFuncDict._unreduce, (self.keyfunc, dict(self), collection_adapter(self)))

    @util.preload_module('sqlalchemy.orm.attributes')
    def _raise_for_unpopulated(self, value: _KT, initiator: Union[AttributeEventToken, Literal[None, False]]=None, *, warn_only: bool) -> None:
        mapper = base.instance_state(value).mapper
        attributes = util.preloaded.orm_attributes
        if not isinstance(initiator, attributes.AttributeEventToken):
            relationship = 'unknown relationship'
        elif initiator.key in mapper.attrs:
            relationship = f'{mapper.attrs[initiator.key]}'
        else:
            relationship = initiator.key
        if warn_only:
            util.warn(f"""Attribute keyed dictionary value for attribute '{relationship}' was None; this will raise in a future release. To skip this assignment entirely, Set the "ignore_unpopulated_attribute=True" parameter on the mapped collection factory.""")
        else:
            raise sa_exc.InvalidRequestError(f"""In event triggered from population of attribute '{relationship}' (potentially from a backref), can't populate value in KeyFuncDict; dictionary key derived from {base.instance_str(value)} is not populated. Ensure appropriate state is set up on the {base.instance_str(value)} object before assigning to the {relationship} attribute. To skip this assignment entirely, Set the "ignore_unpopulated_attribute=True" parameter on the mapped collection factory.""")

    @collection.appender
    @collection.internally_instrumented
    def set(self, value: _KT, _sa_initiator: Union[AttributeEventToken, Literal[None, False]]=None) -> None:
        """Add an item by value, consulting the keyfunc for the key."""
        key = self.keyfunc(value)
        if key is base.NO_VALUE:
            if not self.ignore_unpopulated_attribute:
                self._raise_for_unpopulated(value, _sa_initiator, warn_only=False)
            else:
                return
        elif key is _UNMAPPED_AMBIGUOUS_NONE:
            if not self.ignore_unpopulated_attribute:
                self._raise_for_unpopulated(value, _sa_initiator, warn_only=True)
                key = None
            else:
                return
        self.__setitem__(key, value, _sa_initiator)

    @collection.remover
    @collection.internally_instrumented
    def remove(self, value: _KT, _sa_initiator: Union[AttributeEventToken, Literal[None, False]]=None) -> None:
        """Remove an item by value, consulting the keyfunc for the key."""
        key = self.keyfunc(value)
        if key is base.NO_VALUE:
            if not self.ignore_unpopulated_attribute:
                self._raise_for_unpopulated(value, _sa_initiator, warn_only=False)
            return
        elif key is _UNMAPPED_AMBIGUOUS_NONE:
            if not self.ignore_unpopulated_attribute:
                self._raise_for_unpopulated(value, _sa_initiator, warn_only=True)
                key = None
            else:
                return
        if self[key] != value:
            raise sa_exc.InvalidRequestError("Can not remove '%s': collection holds '%s' for key '%s'. Possible cause: is the KeyFuncDict key function based on mutable properties or properties that only obtain values after flush?" % (value, self[key], key))
        self.__delitem__(key, _sa_initiator)