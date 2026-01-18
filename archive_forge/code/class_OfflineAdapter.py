import io
import os
import threading
import time
import uuid
from functools import lru_cache
from http import HTTPStatus
from typing import Callable, Tuple, Type, Union
import requests
from requests import Response
from requests.adapters import HTTPAdapter
from requests.models import PreparedRequest
from .. import constants
from . import logging
from ._typing import HTTP_METHOD_T
class OfflineAdapter(HTTPAdapter):

    def send(self, request: PreparedRequest, *args, **kwargs) -> Response:
        raise OfflineModeIsEnabled(f'Cannot reach {request.url}: offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable.')