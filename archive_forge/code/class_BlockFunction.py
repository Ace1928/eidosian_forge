from __future__ import annotations
import copy
import hashlib
import inspect
import json
import os
import random
import secrets
import string
import sys
import threading
import time
import warnings
import webbrowser
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Literal, Sequence, cast
from urllib.parse import urlparse, urlunparse
import anyio
import fastapi
import httpx
from anyio import CapacityLimiter
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import (
from gradio.blocks_events import BlocksEvents, BlocksMeta
from gradio.context import Context
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.events import (
from gradio.exceptions import (
from gradio.helpers import create_tracker, skip, special_args
from gradio.state_holder import SessionState
from gradio.themes import Default as DefaultTheme
from gradio.themes import ThemeClass as Theme
from gradio.tunneling import (
from gradio.utils import (
class BlockFunction:

    def __init__(self, fn: Callable | None, inputs: list[Component], outputs: list[Component], preprocess: bool, postprocess: bool, inputs_as_dict: bool, batch: bool=False, max_batch_size: int=4, concurrency_limit: int | None | Literal['default']='default', concurrency_id: str | None=None, tracks_progress: bool=False):
        self.fn = fn
        self.inputs = inputs
        self.outputs = outputs
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.tracks_progress = tracks_progress
        self.concurrency_limit: int | None | Literal['default'] = concurrency_limit
        self.concurrency_id = concurrency_id or str(id(fn))
        self.batch = batch
        self.max_batch_size = max_batch_size
        self.total_runtime = 0
        self.total_runs = 0
        self.inputs_as_dict = inputs_as_dict
        self.name = getattr(fn, '__name__', 'fn') if fn is not None else None
        self.spaces_auto_wrap()

    def spaces_auto_wrap(self):
        if spaces is None:
            return
        if utils.get_space() is None:
            return
        self.fn = spaces.gradio_auto_wrap(self.fn)

    def __str__(self):
        return str({'fn': self.name, 'preprocess': self.preprocess, 'postprocess': self.postprocess})

    def __repr__(self):
        return str(self)