from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
def __reverse(self) -> None:
    self._reversed = True
    lastUrl = self._getLastPageUrl()
    if lastUrl:
        self.__nextUrl = lastUrl