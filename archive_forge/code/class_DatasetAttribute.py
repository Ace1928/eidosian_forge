import typing
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from functools import lru_cache
from numbers import Number
from types import MappingProxyType
from typing import (
from pennylane.data.base import hdf5
from pennylane.data.base.hdf5 import HDF5, HDF5Any, HDF5Group
from pennylane.data.base.typing_util import UNSET, get_type, get_type_str
class DatasetAttribute(ABC, Generic[HDF5, ValueType, InitValueType]):
    """
    The DatasetAttribute class provides an interface for converting Python objects to and from a HDF5
    array or Group. It uses the registry pattern to maintain a mapping of type_id to
    DatasetAttribute, and Python types to compatible AttributeTypes.

    Attributes:
        type_id: Unique identifier for this DatasetAttribute class. Must be declared
            in subclasses.
    """
    type_id: ClassVar[str]

    @abstractmethod
    def hdf5_to_value(self, bind: HDF5) -> ValueType:
        """Parses bind into Python object."""

    @abstractmethod
    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: InitValueType) -> HDF5:
        """Converts value into a HDF5 Array or Group under bind_parent[key]."""

    @overload
    def __init__(self, value: Union[InitValueType, Literal[UNSET]]=UNSET, info: Optional[AttributeInfo]=None, *, parent_and_key: Optional[Tuple[HDF5Group, str]]=None):
        """Initialize a new dataset attribute from ``value``.

        Args:
            value: Value that will be stored in dataset attribute.
            info: Metadata to attach to attribute.
            parent_and_key: A 2-tuple specifying the HDF5 group that will contain
                this attribute, and its key. If None, attribute will be stored in-memory.
        """

    @overload
    def __init__(self, *, bind: HDF5):
        """Load previously persisted dataset attribute from ``bind``.

        If ``bind`` contains an attribute of a different type, or does not
        contain a dataset attribute, a ``TypeError` will be raised.

        Args:
            bind: HDF5 object from which existing attribute will be loaded.
        """

    def __init__(self, value: Union[InitValueType, Literal[UNSET]]=UNSET, info: Optional[AttributeInfo]=None, *, bind: Optional[HDF5]=None, parent_and_key: Optional[Tuple[HDF5Group, str]]=None) -> None:
        """
        Initialize a new dataset attribute, or load from an existing
        hdf5 object.

        This constructor can be called two ways: value initialization
        or bind initialization.

        Value initialization creates the attribute with specified ``value`` in
        a new HDF5 object, with optional ``info`` attached. The attribute can
        be created in an existing HDF5 group by passing the ``parent_and_key``
        argument.

        Bind initialization loads an attribute that was previously persisted
        in HDF5 object ``bind``.

        Note that if ``bind`` is provided, all other arguments will be ignored.

        Args:
            value: Value to initialize attribute to
            info: Metadata to attach to attribute
            bind: HDF5 object from which existing attribute will be loaded
            parent_and_key: A 2-tuple specifying the HDF5 group that will contain
                this attribute, and its key.
        """
        if bind is not None:
            self._bind_init(bind)
        else:
            self._value_init(value, info, parent_and_key)

    def _bind_init(self, bind: HDF5) -> None:
        """Constructor for bind initialization. See __init__()."""
        self._bind = bind
        self._check_bind()

    def _value_init(self, value: Union[InitValueType, Literal[UNSET]], info: Optional[AttributeInfo], parent_and_key: Optional[Tuple[HDF5Group, str]]):
        """Constructor for value initialization. See __init__()."""
        if parent_and_key is not None:
            parent, key = parent_and_key
        else:
            parent, key = (hdf5.create_group(), '_')
        if value is UNSET:
            value = self.default_value()
            if value is UNSET:
                raise TypeError("__init__() missing 1 required positional argument: 'value'")
        self._bind = self._set_value(value, info, parent, key)
        self._check_bind()
        self.__post_init__(value)

    @property
    def info(self) -> AttributeInfo:
        """Returns the ``AttributeInfo`` for this attribute."""
        return AttributeInfo(self.bind.attrs)

    @property
    def bind(self) -> HDF5:
        """Returns the HDF5 object that contains this attribute's
        data."""
        return self._bind

    @classmethod
    def default_value(cls) -> Union[InitValueType, Literal[UNSET]]:
        """Returns a valid default value for this type, or ``UNSET`` if this type
        must be initialized with a value."""
        return UNSET

    @classmethod
    def py_type(cls, value_type: Type[InitValueType]) -> str:
        """Determines the ``py_type`` of an attribute during value initialization,
        if it was not provided in the ``info`` argument. This method returns
        ``f"{value_type.__module__}.{value_type.__name__}``.
        """
        return get_type_str(value_type)

    @classmethod
    def consumes_types(cls) -> typing.Iterable[type]:
        """
        Returns an iterable of types for which this should be the default
        codec. If a value of one of these types is assigned to a Dataset
        without specifying a `type_id`, this type will be used.
        """
        return ()

    def __post_init__(self, value: InitValueType) -> None:
        """Called after __init__(), only during value initialization. Can be implemented
        in subclasses that require additional initialization."""

    def get_value(self) -> ValueType:
        """Deserializes the mapped value from ``bind``."""
        return self.hdf5_to_value(self.bind)

    def copy_value(self) -> ValueType:
        """Deserializes the mapped value from ``bind``, and also perform a 'deep-copy'
        of any nested values contained in ``bind``."""
        return self.get_value()

    def _set_value(self, value: InitValueType, info: Optional[AttributeInfo], parent: HDF5Group, key: str) -> HDF5:
        """Converts ``value`` into HDF5 format and sets the attribute info."""
        if info is None:
            info = AttributeInfo()
        info['type_id'] = self.type_id
        if info.py_type is None:
            info.py_type = self.py_type(type(value))
        new_bind = self.value_to_hdf5(parent, key, value)
        new_info = AttributeInfo(new_bind.attrs)
        info.save(new_info)
        return new_bind

    def _set_parent(self, parent: HDF5Group, key: str):
        """Copies this attribute's data into ``parent``, under ``key``."""
        hdf5.copy(source=self.bind, dest=parent, key=key, on_conflict='overwrite')
        self._bind = parent[key]

    def _check_bind(self):
        """
        Checks that ``bind.attrs`` contains the type_id corresponding to
        this type.
        """
        existing_type_id = self.info.get('type_id')
        if existing_type_id is None:
            raise ValueError("'bind' does not contain a dataset attribute.")
        if existing_type_id != self.type_id:
            raise TypeError(f"'bind' is bound to another attribute type '{existing_type_id}'")

    def __copy__(self) -> 'DatasetAttribute':
        impl_group = hdf5.create_group()
        hdf5.copy(self.bind, impl_group, '_')
        return type(self)(bind=impl_group['_'])

    def __deepcopy__(self, memo) -> 'DatasetAttribute':
        return self.__copy__()

    def __eq__(self, __value: object) -> bool:
        return self.get_value() == __value

    def __repr__(self) -> str:
        return f'{type(self).__name__}({repr(self.get_value())})'

    def __str__(self) -> str:
        return str(self.get_value())
    __registry: typing.Mapping[str, Type['DatasetAttribute']] = {}
    __type_consumer_registry: typing.Mapping[type, Type['DatasetAttribute']] = {}
    registry: typing.Mapping[str, Type['DatasetAttribute']] = MappingProxyType(__registry)
    'Maps type_ids to their DatasetAttribute classes.'
    type_consumer_registry: typing.Mapping[type, Type['DatasetAttribute']] = MappingProxyType(__type_consumer_registry)
    'Maps types to their default DatasetAttribute'

    def __init_subclass__(cls, *, abstract: bool=False) -> None:
        if abstract:
            return super().__init_subclass__()
        existing_type = DatasetAttribute.__registry.get(cls.type_id)
        if existing_type is not None:
            raise TypeError(f"DatasetAttribute with type_id '{cls.type_id}' already exists: {existing_type}")
        DatasetAttribute.__registry[cls.type_id] = cls
        for type_ in cls.consumes_types():
            existing_type = DatasetAttribute.type_consumer_registry.get(type_)
            if existing_type is not None:
                warnings.warn(f"Conflicting default types: Both '{cls.__name__}' and '{existing_type.__name__}' consume type '{type_.__name__}'. '{type_.__name__}' will now be consumed by '{cls.__name__}'")
            DatasetAttribute.__type_consumer_registry[type_] = cls
        return super().__init_subclass__()