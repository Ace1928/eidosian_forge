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
def _tracing_thread_drain_queue(tracing_queue: Queue, limit: int=100, block: bool=True) -> List[TracingQueueItem]:
    next_batch: List[TracingQueueItem] = []
    try:
        if (item := tracing_queue.get(block=block, timeout=0.25)):
            next_batch.append(item)
        while (item := tracing_queue.get(block=block, timeout=0.05)):
            next_batch.append(item)
            if limit and len(next_batch) >= limit:
                break
    except Empty:
        pass
    return next_batch