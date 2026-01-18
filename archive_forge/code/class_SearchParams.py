import abc
import time
import random
import asyncio
import contextlib
from pydantic import BaseModel
from lazyops.imports._aiohttpx import resolve_aiohttpx
from lazyops.imports._bs4 import resolve_bs4
from urllib.parse import quote_plus
import aiohttpx
from bs4 import BeautifulSoup, Tag
from .utils import filter_result, load_user_agents, load_cookie_jar, get_random_jitter
from typing import List, Optional, Dict, Any, Union, Tuple, Set, Callable, Awaitable, TypeVar, Generator, AsyncGenerator
class SearchParams(BaseModel):
    """
    The search params
    """
    query: str
    tld: Optional[str] = 'com'
    lang: Optional[str] = 'en'
    tbs: Optional[str] = '0'
    safe: Optional[str] = 'off'
    num: Optional[int] = 10
    start: Optional[int] = 0
    stop: Optional[int] = None
    country: Optional[str] = ''
    include_title: Optional[bool] = False
    include_description: Optional[bool] = False

    @property
    def url_home(self) -> str:
        """
        Gets the home url
        """
        return f'https://www.google.{self.tld}/'

    @property
    def url_search(self) -> str:
        """
        Gets the search url
        """
        return f'https://www.google.{self.tld}/search?lr=lang_{self.lang}&q={self.query}&btnG=Google+Search&tbs={self.tbs}&safe={self.safe}&cr={self.country}&filter=0'

    @property
    def url_next_page(self) -> str:
        """
        Gets the next page url
        """
        return f'https://www.google.{self.tld}/search?lr=lang_{self.lang}&q={self.query}&start={self.start}&tbs={self.tbs}&safe={self.safe}&cr={self.country}&filter=0'

    @property
    def url_search_num(self) -> str:
        """
        Gets the search url
        """
        return f'https://www.google.{self.tld}/search?lr=lang_{self.lang}&q={self.query}&num={self.num}&btnG=Google+Search&tbs={self.tbs}&&safe={self.safe}scr={self.country}&filter=0'

    @property
    def url_next_page_num(self) -> str:
        """
        Gets the next page url
        """
        return f'https://www.google.{self.tld}/search?lr=lang_{self.lang}&q={self.query}&num={self.num}&start={self.start}&tbs={self.tbs}&safe={self.safe}&cr={self.country}&filter=0'

    @property
    def start_url(self) -> str:
        """
        Gets the start url
        """
        if self.start:
            return self.url_next_page if self.num == 10 else self.url_next_page_num
        return self.url_search if self.num == 10 else self.url_search_num

    def get_next_url(self) -> str:
        """
        Gets the next url
        """
        self.start += self.num
        return self.url_next_page if self.num == 10 else self.url_next_page_num

    def is_valid_link(self, anchor: Tag, hashes: Set) -> Optional[str]:
        """
        Validates the link
        """
        try:
            link = anchor['href']
        except KeyError:
            return None
        link = filter_result(link)
        if not link:
            return None
        h = hash(link)
        if h in hashes:
            return None
        hashes.add(h)
        return link

    def extract_result(self, anchor: Tag, desc: Tag, hashes: Set) -> Optional[RT]:
        """
        Extracts the result
        """
        link = self.is_valid_link(anchor, hashes)
        if not link:
            return None
        if self.include_title and self.include_description:
            return (link, anchor.text, desc.text)
        if self.include_description:
            return (link, desc.text)
        return (link, anchor.text) if self.include_title else link