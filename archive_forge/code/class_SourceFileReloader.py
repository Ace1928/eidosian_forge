from __future__ import annotations
import ast
import asyncio
import copy
import dataclasses
import functools
import importlib
import importlib.util
import inspect
import json
import json.decoder
import os
import pkgutil
import re
import sys
import tempfile
import threading
import time
import traceback
import typing
import urllib.parse
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from io import BytesIO
from numbers import Number
from pathlib import Path
from types import AsyncGeneratorType, GeneratorType, ModuleType
from typing import (
import anyio
import gradio_client.utils as client_utils
import httpx
from gradio_client.documentation import document
from typing_extensions import ParamSpec
import gradio
from gradio.context import Context
from gradio.data_classes import FileData
from gradio.strings import en
class SourceFileReloader(BaseReloader):

    def __init__(self, app: App, watch_dirs: list[str], watch_module_name: str, demo_file: str, stop_event: threading.Event, change_event: threading.Event, demo_name: str='demo') -> None:
        super().__init__()
        self.app = app
        self.watch_dirs = watch_dirs
        self.watch_module_name = watch_module_name
        self.stop_event = stop_event
        self.change_event = change_event
        self.demo_name = demo_name
        self.demo_file = Path(demo_file)

    @property
    def running_app(self) -> App:
        return self.app

    def should_watch(self) -> bool:
        return not self.stop_event.is_set()

    def stop(self) -> None:
        self.stop_event.set()

    def alert_change(self):
        self.change_event.set()

    def swap_blocks(self, demo: Blocks):
        super().swap_blocks(demo)
        self.alert_change()