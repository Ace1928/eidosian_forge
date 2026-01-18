from typing import Any, Callable, cast, List, Optional
from types import ModuleType
from importlib.machinery import ModuleSpec
from importlib.abc import Loader
from contextlib import contextmanager
import importlib
from importlib import abc
import sys
Load the module and insert it into the parent's globals.