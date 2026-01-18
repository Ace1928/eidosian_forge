from __future__ import annotations
import functools
import sys
import typing
import warnings
import anyio.to_thread
class _StopIteration(Exception):
    pass