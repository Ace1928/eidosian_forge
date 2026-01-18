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
def _tracing_sub_thread_func(client_ref: weakref.ref[Client]) -> None:
    client = client_ref()
    if client is None:
        return
    try:
        if not client.info:
            return
    except BaseException as e:
        logger.debug('Error in tracing control thread: %s', e)
        return
    tracing_queue = client.tracing_queue
    assert tracing_queue is not None
    batch_ingest_config = _ensure_ingest_config(client.info)
    size_limit = batch_ingest_config.get('size_limit', 100)
    seen_successive_empty_queues = 0
    while threading.main_thread().is_alive() and seen_successive_empty_queues <= batch_ingest_config['scale_down_nempty_trigger']:
        if (next_batch := _tracing_thread_drain_queue(tracing_queue, limit=size_limit)):
            seen_successive_empty_queues = 0
            _tracing_thread_handle_batch(client, tracing_queue, next_batch)
        else:
            seen_successive_empty_queues += 1
    while (next_batch := _tracing_thread_drain_queue(tracing_queue, limit=size_limit, block=False)):
        _tracing_thread_handle_batch(client, tracing_queue, next_batch)