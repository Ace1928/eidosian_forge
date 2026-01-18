import configparser
import contextlib
import locale
import logging
import pathlib
import re
import sys
from itertools import chain, groupby, repeat
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union
from pip._vendor.requests.models import Request, Response
from pip._vendor.rich.console import Console, ConsoleOptions, RenderResult
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text
class NetworkConnectionError(PipError):
    """HTTP connection error"""

    def __init__(self, error_msg: str, response: Optional[Response]=None, request: Optional[Request]=None) -> None:
        """
        Initialize NetworkConnectionError with  `request` and `response`
        objects.
        """
        self.response = response
        self.request = request
        self.error_msg = error_msg
        if self.response is not None and (not self.request) and hasattr(response, 'request'):
            self.request = self.response.request
        super().__init__(error_msg, response, request)

    def __str__(self) -> str:
        return str(self.error_msg)