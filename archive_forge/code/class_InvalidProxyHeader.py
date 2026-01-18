import contextlib
from typing import Callable, Generator, Type
class InvalidProxyHeader(Exception):
    """
    The provided PROXY protocol header is invalid.
    """