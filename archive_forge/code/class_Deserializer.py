from __future__ import annotations
import logging # isort:skip
import base64
import datetime as dt
import sys
from array import array as TypedArray
from math import isinf, isnan
from types import SimpleNamespace
from typing import (
import numpy as np
from ..util.dataclasses import (
from ..util.dependencies import uses_pandas
from ..util.serialization import (
from ..util.warnings import BokehUserWarning, warn
from .types import ID
class Deserializer:
    """ Convert from serializable representations to built-in and custom types. """
    _decoders: ClassVar[dict[str, Decoder]] = {}

    @classmethod
    def register(cls, type: str, decoder: Decoder) -> None:
        assert type not in cls._decoders, f"'{type} is already registered"
        cls._decoders[type] = decoder
    _references: dict[ID, Model]
    _setter: Setter | None
    _decoding: bool
    _buffers: dict[ID, Buffer]

    def __init__(self, references: Sequence[Model] | None=None, *, setter: Setter | None=None):
        self._references = {obj.id: obj for obj in references or []}
        self._setter = setter
        self._decoding = False
        self._buffers = {}

    def has_ref(self, obj: Model) -> bool:
        return obj.id in self._references

    def deserialize(self, obj: Any | Serialized[Any]) -> Any:
        if isinstance(obj, Serialized):
            return self.decode(obj.content, obj.buffers)
        else:
            return self.decode(obj)

    def decode(self, obj: AnyRep, buffers: list[Buffer] | None=None) -> Any:
        if buffers is not None:
            for buffer in buffers:
                self._buffers[buffer.id] = buffer
        if self._decoding:
            return self._decode(obj)
        self._decoding = True
        try:
            return self._decode(obj)
        finally:
            self._buffers.clear()
            self._decoding = False

    def _decode(self, obj: AnyRep) -> Any:
        if isinstance(obj, dict):
            if 'type' in obj:
                type = obj['type']
                if type in self._decoders:
                    return self._decoders[type](obj, self)
                elif type == 'ref':
                    return self._decode_ref(cast(Ref, obj))
                elif type == 'symbol':
                    return self._decode_symbol(cast(SymbolRep, obj))
                elif type == 'number':
                    return self._decode_number(cast(NumberRep, obj))
                elif type == 'array':
                    return self._decode_array(cast(ArrayRep, obj))
                elif type == 'set':
                    return self._decode_set(cast(SetRep, obj))
                elif type == 'map':
                    return self._decode_map(cast(MapRep, obj))
                elif type == 'bytes':
                    return self._decode_bytes(cast(BytesRep, obj))
                elif type == 'slice':
                    return self._decode_slice(cast(SliceRep, obj))
                elif type == 'typed_array':
                    return self._decode_typed_array(cast(TypedArrayRep, obj))
                elif type == 'ndarray':
                    return self._decode_ndarray(cast(NDArrayRep, obj))
                elif type == 'object':
                    if 'id' in obj:
                        return self._decode_object_ref(cast(ObjectRefRep, obj))
                    else:
                        return self._decode_object(cast(ObjectRep, obj))
                else:
                    self.error(f"unable to decode an object of type '{type}'")
            elif 'id' in obj:
                return self._decode_ref(cast(Ref, obj))
            else:
                return {key: self._decode(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [self._decode(entry) for entry in obj]
        else:
            return obj

    def _decode_ref(self, obj: Ref) -> Model:
        id = obj['id']
        instance = self._references.get(id)
        if instance is not None:
            return instance
        else:
            self.error(UnknownReferenceError(id))

    def _decode_symbol(self, obj: SymbolRep) -> float:
        name = obj['name']
        self.error(f"can't resolve named symbol '{name}'")

    def _decode_number(self, obj: NumberRep) -> float:
        value = obj['value']
        return float(value) if isinstance(value, str) else value

    def _decode_array(self, obj: ArrayRep) -> list[Any]:
        entries = obj.get('entries', [])
        return [self._decode(entry) for entry in entries]

    def _decode_set(self, obj: SetRep) -> set[Any]:
        entries = obj.get('entries', [])
        return {self._decode(entry) for entry in entries}

    def _decode_map(self, obj: MapRep) -> dict[Any, Any]:
        entries = obj.get('entries', [])
        return {self._decode(key): self._decode(val) for key, val in entries}

    def _decode_bytes(self, obj: BytesRep) -> bytes:
        data = obj['data']
        if isinstance(data, str):
            return base64.b64decode(data)
        elif isinstance(data, Buffer):
            buffer = data
        else:
            id = data['id']
            if id in self._buffers:
                buffer = self._buffers[id]
            else:
                self.error(f"can't resolve buffer '{id}'")
        return buffer.data

    def _decode_slice(self, obj: SliceRep) -> slice:
        start = self._decode(obj['start'])
        stop = self._decode(obj['stop'])
        step = self._decode(obj['step'])
        return slice(start, stop, step)

    def _decode_typed_array(self, obj: TypedArrayRep) -> TypedArray[Any]:
        array = obj['array']
        order = obj['order']
        dtype = obj['dtype']
        data = self._decode(array)
        dtype_to_typecode = dict(uint8='B', int8='b', uint16='H', int16='h', uint32='I', int32='i', float32='f', float64='d')
        typecode = dtype_to_typecode.get(dtype)
        if typecode is None:
            self.error(f"unsupported dtype '{dtype}'")
        typed_array: TypedArray[Any] = TypedArray(typecode, data)
        if order != sys.byteorder:
            typed_array.byteswap()
        return typed_array

    def _decode_ndarray(self, obj: NDArrayRep) -> npt.NDArray[Any]:
        array = obj['array']
        order = obj['order']
        dtype = obj['dtype']
        shape = obj['shape']
        decoded = self._decode(array)
        ndarray: npt.NDArray[Any]
        if isinstance(decoded, bytes):
            ndarray = np.copy(np.frombuffer(decoded, dtype=dtype))
            if order != sys.byteorder:
                ndarray.byteswap(inplace=True)
        else:
            ndarray = np.array(decoded, dtype=dtype)
        if len(shape) > 1:
            ndarray = ndarray.reshape(shape)
        return ndarray

    def _decode_object(self, obj: ObjectRep) -> object:
        raise NotImplementedError()

    def _decode_object_ref(self, obj: ObjectRefRep) -> Model:
        id = obj['id']
        instance = self._references.get(id)
        if instance is not None:
            warn(f"reference already known '{id}'", BokehUserWarning)
            return instance
        name = obj['name']
        attributes = obj.get('attributes')
        cls = self._resolve_type(name)
        instance = cls.__new__(cls, id=id)
        if instance is None:
            self.error(f"can't instantiate {name}(id={id})")
        self._references[instance.id] = instance
        if not instance._initialized:
            from .has_props import HasProps
            HasProps.__init__(instance)
        if attributes is not None:
            decoded_attributes = {key: self._decode(val) for key, val in attributes.items()}
            for key, val in decoded_attributes.items():
                instance.set_from_json(key, val, setter=self._setter)
        return instance

    def _resolve_type(self, type: str) -> type[Model]:
        from ..model import Model
        cls = Model.model_class_reverse_map.get(type)
        if cls is not None:
            if issubclass(cls, Model):
                return cls
            else:
                self.error(f"object of type '{type}' is not a subclass of 'Model'")
        elif type == 'Figure':
            from ..plotting import figure
            return figure
        else:
            self.error(f"can't resolve type '{type}'")

    def error(self, error: str | DeserializationError) -> NoReturn:
        if isinstance(error, str):
            raise DeserializationError(error)
        else:
            raise error