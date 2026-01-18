import contextlib
from typing import Iterator, Mapping, Type
class ConnectionNotAvailable(Exception):
    pass