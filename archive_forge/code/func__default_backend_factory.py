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
def _default_backend_factory() -> requests.Session:
    session = requests.Session()
    if constants.HF_HUB_OFFLINE:
        session.mount('http://', OfflineAdapter())
        session.mount('https://', OfflineAdapter())
    else:
        session.mount('http://', UniqueRequestIdAdapter())
        session.mount('https://', UniqueRequestIdAdapter())
    return session