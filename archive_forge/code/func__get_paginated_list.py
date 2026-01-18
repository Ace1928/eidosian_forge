from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def _get_paginated_list(self, path: str, *, params: Optional[dict]=None) -> Iterator[dict]:
    """Get a paginated list of items.

        Parameters
        ----------
        path : str
            The path of the request URL.
        params : dict or None, default=None
            The query parameters.

        Yields:
        ------
        dict
            The items in the paginated list.
        """
    params_ = params.copy() if params else {}
    offset = params_.get('offset', 0)
    params_['limit'] = params_.get('limit', 100)
    while True:
        params_['offset'] = offset
        response = self.request_with_retries('GET', path, params=params_)
        items = response.json()
        if not items:
            break
        yield from items
        if len(items) < params_['limit']:
            break
        offset += len(items)