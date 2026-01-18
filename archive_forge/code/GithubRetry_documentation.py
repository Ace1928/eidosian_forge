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

        :param secondary_rate_wait: seconds to wait before retrying secondary rate limit errors
        :param kwargs: see urllib3.Retry for more arguments
        