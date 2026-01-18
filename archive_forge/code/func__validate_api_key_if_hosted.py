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
def _validate_api_key_if_hosted(api_url: str, api_key: Optional[str]) -> None:
    """Verify API key is provided if url not localhost.

    Parameters
    ----------
    api_url : str
        The API URL.
    api_key : str or None
        The API key.

    Raises:
    ------
    LangSmithUserError
        If the API key is not provided when using the hosted service.
    """
    if not api_key:
        if _is_langchain_hosted(api_url):
            raise ls_utils.LangSmithUserError('API key must be provided when using hosted LangSmith API')