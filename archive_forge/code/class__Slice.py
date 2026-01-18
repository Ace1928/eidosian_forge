from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
class _Slice:

    def __init__(self, theList: 'PaginatedListBase[T]', theSlice: slice):
        self.__list = theList
        self.__start = theSlice.start or 0
        self.__stop = theSlice.stop
        self.__step = theSlice.step or 1

    def __iter__(self) -> Iterator[T]:
        index = self.__start
        while not self.__finished(index):
            if self.__list._isBiggerThan(index):
                yield self.__list[index]
                index += self.__step
            else:
                return

    def __finished(self, index: int) -> bool:
        return self.__stop is not None and index >= self.__stop