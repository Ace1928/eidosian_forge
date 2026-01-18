from abc import abstractmethod
from typing import Any, Callable, Union
import re
import codecs
from ..helpers import collapse_white_spaces
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
class AbstractBinary(AnyAtomicType):
    """
    Abstract class for xs:base64Binary data.

    :param value: a string or a binary data or an untyped atomic instance.
    :param ordered: a boolean that enable total ordering for the instance, `False` for default.
    """
    value: bytes
    invalid_type: Callable[[Any], TypeError]

    def __init__(self, value: Union[str, bytes, UntypedAtomic, 'AbstractBinary'], ordered: bool=False) -> None:
        self.ordered = ordered
        if isinstance(value, self.__class__):
            self.value = value.value
        elif isinstance(value, AbstractBinary):
            self.value = self.encoder(value.decode())
        else:
            if isinstance(value, UntypedAtomic):
                value = collapse_white_spaces(value.value)
            elif isinstance(value, str):
                value = collapse_white_spaces(value)
            elif isinstance(value, bytes):
                value = collapse_white_spaces(value.decode('utf-8'))
            else:
                raise self.invalid_type(value)
            self.validate(value)
            self.value = value.replace(' ', '').encode('ascii')

    def __repr__(self) -> str:
        return '%s(%r)' % (self.__class__.__name__, self.value)

    def __bytes__(self) -> bytes:
        return self.value

    @classmethod
    def validate(cls, value: object) -> None:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def encoder(value: bytes) -> bytes:
        raise NotImplementedError()

    @abstractmethod
    def decode(self) -> bytes:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AbstractBinary):
            return self.decode() == other.decode()
        else:
            return NotImplemented

    def __lt__(self, other: object) -> bool:
        if not self.ordered or not isinstance(other, AbstractBinary):
            return NotImplemented
        for oct1, oct2 in zip(self.decode(), other.decode()):
            if oct1 != oct2:
                return oct1 < oct2
        return len(self.decode()) < len(other.decode())

    def __le__(self, other: object) -> bool:
        if not self.ordered or not isinstance(other, AbstractBinary):
            return NotImplemented
        for oct1, oct2 in zip(self.decode(), other.decode()):
            if oct1 != oct2:
                return oct1 < oct2
        return len(self.decode()) <= len(other.decode())

    def __gt__(self, other: object) -> bool:
        if not self.ordered or not isinstance(other, AbstractBinary):
            return NotImplemented
        for oct1, oct2 in zip(self.decode(), other.decode()):
            if oct1 != oct2:
                return oct1 > oct2
        return len(self.decode()) > len(other.decode())

    def __ge__(self, other: object) -> bool:
        if not self.ordered or not isinstance(other, AbstractBinary):
            return NotImplemented
        for oct1, oct2 in zip(self.decode(), other.decode()):
            if oct1 != oct2:
                return oct1 > oct2
        return len(self.decode()) >= len(other.decode())