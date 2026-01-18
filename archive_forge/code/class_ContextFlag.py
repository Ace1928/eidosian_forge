import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
class ContextFlag:
    """A context manager which is also used as a boolean flag value within the default sigint handler.

    Its main use is as a flag to prevent the SIGINT handler in cmd2 from raising a KeyboardInterrupt
    while a critical code section has set the flag to True. Because signal handling is always done on the
    main thread, this class is not thread-safe since there is no need.
    """

    def __init__(self) -> None:
        self.__count = 0

    def __bool__(self) -> bool:
        return self.__count > 0

    def __enter__(self) -> None:
        self.__count += 1

    def __exit__(self, *args: Any) -> None:
        self.__count -= 1
        if self.__count < 0:
            raise ValueError('count has gone below 0')