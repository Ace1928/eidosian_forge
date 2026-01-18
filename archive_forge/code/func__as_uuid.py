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
def _as_uuid(value: ID_TYPE, var: Optional[str]=None) -> uuid.UUID:
    try:
        return uuid.UUID(value) if not isinstance(value, uuid.UUID) else value
    except ValueError as e:
        var = var or 'value'
        raise ls_utils.LangSmithUserError(f'{var} must be a valid UUID or UUID string. Got {value}') from e