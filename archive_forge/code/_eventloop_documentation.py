from __future__ import annotations
import math
import sys
import threading
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any, TypeVar
import sniffio
Return the current async library's cancellation exception class.