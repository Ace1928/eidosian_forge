import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sized
from http.cookies import BaseCookie, Morsel
from typing import (
from multidict import CIMultiDict
from yarl import URL
from .helpers import get_running_loop
from .typedefs import LooseCookies
class AbstractRouter(ABC):

    def __init__(self) -> None:
        self._frozen = False

    def post_init(self, app: Application) -> None:
        """Post init stage.

        Not an abstract method for sake of backward compatibility,
        but if the router wants to be aware of the application
        it can override this.
        """

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Freeze router."""
        self._frozen = True

    @abstractmethod
    async def resolve(self, request: Request) -> 'AbstractMatchInfo':
        """Return MATCH_INFO for given request"""