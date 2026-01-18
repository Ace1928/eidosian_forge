from __future__ import annotations
import asyncio
import copy
import os
import random
import time
import traceback
import uuid
from collections import defaultdict
from queue import Queue as ThreadQueue
from typing import TYPE_CHECKING
import fastapi
from typing_extensions import Literal
from gradio import route_utils, routes
from gradio.data_classes import (
from gradio.exceptions import Error
from gradio.helpers import TrackedIterable
from gradio.server_messages import (
from gradio.utils import LRUCache, run_coro_in_background, safe_get_lock, set_task_name
def _cancel_asyncio_tasks(self):
    for task in self._asyncio_tasks:
        task.cancel()
    self._asyncio_tasks = []