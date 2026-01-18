from abc import abstractmethod
from typing import Any, Callable, Union
import re
import codecs
from ..helpers import collapse_white_spaces
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
class HexBinary(AbstractBinary):
    name = 'hexBinary'
    pattern = re.compile('^([0-9a-fA-F]{2})*$')

    @classmethod
    def validate(cls, value: object) -> None:
        if isinstance(value, cls):
            return
        elif isinstance(value, bytes):
            value = value.decode()
        elif not isinstance(value, str):
            raise cls.invalid_type(value)
        value = value.strip()
        if cls.pattern.match(value) is None:
            raise cls.invalid_value(value)

    @staticmethod
    def encoder(value: bytes) -> bytes:
        return codecs.encode(value, 'hex')

    def decode(self) -> bytes:
        return codecs.decode(self.value, 'hex')

    def __str__(self) -> str:
        return self.value.decode('utf-8').upper()

    def __hash__(self) -> int:
        return hash(self.value.upper())

    def __len__(self) -> int:
        return len(self.value) // 2