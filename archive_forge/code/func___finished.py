from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
def __finished(self, index: int) -> bool:
    return self.__stop is not None and index >= self.__stop