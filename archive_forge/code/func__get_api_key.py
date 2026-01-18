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
def _get_api_key(api_key: Optional[str]) -> Optional[str]:
    api_key_ = api_key if api_key is not None else _get_env(('LANGSMITH_API_KEY', 'LANGCHAIN_API_KEY'))
    if api_key_ is None or not api_key_.strip():
        return None
    return api_key_.strip().strip('"').strip("'")