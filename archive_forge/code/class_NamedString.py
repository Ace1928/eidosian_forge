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
class NamedString(str):
    """
    Subclass of str that includes a .name attribute equal to the value of the string itself. This class is used when returning
    a value from the `.preprocess()` methods of the File and UploadButton components. Before Gradio 4.0, these methods returned a file
    object which was then converted to a string filepath using the `.name` attribute. In Gradio 4.0, these methods now return a str
    filepath directly, but to maintain backwards compatibility, we use this class instead of a regular str.
    """

    def __init__(self, *args):
        super().__init__()
        self.name = str(self) if args else ''