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
class ProcessTime:

    def __init__(self):
        self.process_time = 0
        self.count = 0
        self.avg_time = 0

    def add(self, time: float):
        self.process_time += time
        self.count += 1
        self.avg_time = self.process_time / self.count