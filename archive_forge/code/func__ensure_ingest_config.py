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
def _ensure_ingest_config(info: ls_schemas.LangSmithInfo) -> ls_schemas.BatchIngestConfig:
    default_config = ls_schemas.BatchIngestConfig(size_limit_bytes=None, size_limit=100, scale_up_nthreads_limit=_AUTO_SCALE_UP_NTHREADS_LIMIT, scale_up_qsize_trigger=_AUTO_SCALE_UP_QSIZE_TRIGGER, scale_down_nempty_trigger=_AUTO_SCALE_DOWN_NEMPTY_TRIGGER)
    if not info:
        return default_config
    try:
        if not info.batch_ingest_config:
            return default_config
        return info.batch_ingest_config
    except BaseException:
        return default_config