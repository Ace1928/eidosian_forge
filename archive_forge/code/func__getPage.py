from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester
def _getPage(self, data: Any, headers: Dict[str, Any]) -> List[T]:
    self.__nextUrl = None
    if len(data) > 0:
        links = self.__parseLinkHeader(headers)
        if self._reversed:
            if 'prev' in links:
                self.__nextUrl = links['prev']
        elif 'next' in links:
            self.__nextUrl = links['next']
    self.__nextParams = None
    if self.__list_item in data:
        self.__totalCount = data.get(self.__total_count_item)
        data = data[self.__list_item]
    content = [self.__contentClass(self.__requester, headers, self._transformAttributes(element), completed=False) for element in data if element is not None]
    if self._reversed:
        return content[::-1]
    return content