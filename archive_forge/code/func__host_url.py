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
@property
def _host_url(self) -> str:
    """The web host url."""
    if self._web_url:
        link = self._web_url
    else:
        parsed_url = urllib_parse.urlparse(self.api_url)
        if _is_localhost(self.api_url):
            link = 'http://localhost'
        elif parsed_url.path.endswith('/api'):
            new_path = parsed_url.path.rsplit('/api', 1)[0]
            link = urllib_parse.urlunparse(parsed_url._replace(path=new_path))
        elif parsed_url.netloc.startswith('dev.'):
            link = 'https://dev.smith.langchain.com'
        else:
            link = 'https://smith.langchain.com'
    return link