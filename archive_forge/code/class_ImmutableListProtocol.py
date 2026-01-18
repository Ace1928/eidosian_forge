import typing
from typing_extensions import Protocol
class ImmutableListProtocol(Protocol[T]):
    """A protocol used in cases where a list is returned, but should not be
    mutated.

    This provides all of the methods of a Sequence, as well as copy(). copy()
    returns a list, which allows mutation as it's a copy and that's (hopefully)
    safe.

    One particular case this is important is for cached values, since python is
    a pass-by-reference language.
    """

    def __iter__(self) -> typing.Iterator[T]:
        ...

    @typing.overload
    def __getitem__(self, index: int) -> T:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> typing.List[T]:
        ...

    def __contains__(self, item: T) -> bool:
        ...

    def __reversed__(self) -> typing.Iterator[T]:
        ...

    def __len__(self) -> int:
        ...

    def __add__(self, other: typing.List[T]) -> typing.List[T]:
        ...

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __ne__(self, other: typing.Any) -> bool:
        ...

    def __le__(self, other: typing.Any) -> bool:
        ...

    def __lt__(self, other: typing.Any) -> bool:
        ...

    def __gt__(self, other: typing.Any) -> bool:
        ...

    def __ge__(self, other: typing.Any) -> bool:
        ...

    def count(self, item: T) -> int:
        ...

    def index(self, item: T) -> int:
        ...

    def copy(self) -> typing.List[T]:
        ...