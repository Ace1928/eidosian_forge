from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
def _getLastPageUrl(self) -> Optional[str]:
    headers, data = self.__requester.requestJsonAndCheck('GET', self.__firstUrl, parameters=self.__nextParams, headers=self.__headers)
    links = self.__parseLinkHeader(headers)
    return links.get('last')