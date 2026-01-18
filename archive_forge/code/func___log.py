import json
import logging
from datetime import datetime, timezone
from logging import Logger
from types import TracebackType
from typing import Any, Optional
from requests import Response
from requests.models import CaseInsensitiveDict
from requests.utils import get_encoding_from_headers
from typing_extensions import Self
from urllib3 import Retry
from urllib3.connectionpool import ConnectionPool
from urllib3.exceptions import MaxRetryError
from urllib3.response import HTTPResponse
from github.GithubException import GithubException
from github.Requester import Requester
def __log(self, level: int, message: str, **kwargs: Any) -> None:
    if self.__logger is None:
        self.__logger = logging.getLogger(__name__)
    if self.__logger.isEnabledFor(level):
        self.__logger.log(level, message, **kwargs)