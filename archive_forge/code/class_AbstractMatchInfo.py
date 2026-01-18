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
class AbstractMatchInfo(ABC):

    @property
    @abstractmethod
    def handler(self) -> Callable[[Request], Awaitable[StreamResponse]]:
        """Execute matched request handler"""

    @property
    @abstractmethod
    def expect_handler(self) -> Callable[[Request], Awaitable[Optional[StreamResponse]]]:
        """Expect handler for 100-continue processing"""

    @property
    @abstractmethod
    def http_exception(self) -> Optional[HTTPException]:
        """HTTPException instance raised on router's resolving, or None"""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return a dict with additional info useful for introspection"""

    @property
    @abstractmethod
    def apps(self) -> Tuple[Application, ...]:
        """Stack of nested applications.

        Top level application is left-most element.

        """

    @abstractmethod
    def add_app(self, app: Application) -> None:
        """Add application to the nested apps stack."""

    @abstractmethod
    def freeze(self) -> None:
        """Freeze the match info.

        The method is called after route resolution.

        After the call .add_app() is forbidden.

        """