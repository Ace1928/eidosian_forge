import collections.abc
import io
import itertools
import types
import typing
class NormalizedType(typing.NamedTuple):
    """
    Normalized type, made it possible to compare, hash between types.
    """
    origin: Type
    args: typing.Union[tuple, frozenset] = tuple()

    def __eq__(self, other):
        if isinstance(other, NormalizedType):
            if self.origin != other.origin:
                return False
            if isinstance(self.args, frozenset) and isinstance(other.args, frozenset):
                return self.args <= other.args and other.args <= self.args
            return self.origin == other.origin and self.args == other.args
        if not self.args:
            return self.origin == other
        return False

    def __hash__(self) -> int:
        if not self.args:
            return hash(self.origin)
        return hash((self.origin, self.args))

    def __repr__(self):
        if not self.args:
            return f'{self.origin}'
        return f'{self.origin}[{self.args}])'