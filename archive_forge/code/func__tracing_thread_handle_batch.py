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
def _tracing_thread_handle_batch(client: Client, tracing_queue: Queue, batch: List[TracingQueueItem]) -> None:
    create = [it.item for it in batch if it.action == 'create']
    update = [it.item for it in batch if it.action == 'update']
    try:
        client.batch_ingest_runs(create=create, update=update, pre_sampled=True)
    except Exception:
        logger.error('Error in tracing queue', exc_info=True)
        pass
    finally:
        for _ in batch:
            tracing_queue.task_done()