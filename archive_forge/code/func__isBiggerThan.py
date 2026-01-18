from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
def _isBiggerThan(self, index: int) -> bool:
    return len(self.__elements) > index or self._couldGrow()