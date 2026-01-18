import os
import textwrap
from enum import auto, Enum
from traceback import extract_stack, format_exc, format_list, StackSummary
from typing import cast, NoReturn, Optional
import torch._guards
from . import config
from .config import is_fbcode
from .utils import counters
import logging
class InvalidBackend(TorchDynamoException):

    def __init__(self, name):
        super().__init__(f'Invalid backend: {name!r}, see `torch._dynamo.list_backends()` for available backends.')