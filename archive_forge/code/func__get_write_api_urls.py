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
def _get_write_api_urls(_write_api_urls: Optional[Dict[str, str]]) -> Dict[str, str]:
    _write_api_urls = _write_api_urls or json.loads(os.getenv('LANGSMITH_RUNS_ENDPOINTS', '{}'))
    processed_write_api_urls = {}
    for url, api_key in _write_api_urls.items():
        processed_url = url.strip()
        if not processed_url:
            raise ls_utils.LangSmithUserError('LangSmith runs API URL within LANGSMITH_RUNS_ENDPOINTS cannot be empty')
        processed_url = processed_url.strip().strip('"').strip("'").rstrip('/')
        processed_api_key = api_key.strip().strip('"').strip("'")
        _validate_api_key_if_hosted(processed_url, processed_api_key)
        processed_write_api_urls[processed_url] = processed_api_key
    return processed_write_api_urls