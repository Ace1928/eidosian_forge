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
def _get_cursor_paginated_list(self, path: str, *, body: Optional[dict]=None, request_method: Literal['GET', 'POST']='POST', data_key: str='runs') -> Iterator[dict]:
    """Get a cursor paginated list of items.

        Parameters
        ----------
        path : str
            The path of the request URL.
        body : dict or None, default=None
            The query body.
        request_method : str, default="post"
            The HTTP request method.
        data_key : str, default="runs"

        Yields:
        ------
        dict
            The items in the paginated list.
        """
    params_ = body.copy() if body else {}
    while True:
        response = self.request_with_retries(request_method, path, request_kwargs={'data': _dumps_json(params_)})
        response_body = response.json()
        if not response_body:
            break
        if not response_body.get(data_key):
            break
        yield from response_body[data_key]
        cursors = response_body.get('cursors')
        if not cursors:
            break
        if not cursors.get('next'):
            break
        params_['cursor'] = cursors['next']