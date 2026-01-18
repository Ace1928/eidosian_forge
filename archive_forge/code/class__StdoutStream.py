import email.message
import logging
import pathlib
import traceback
import urllib.parse
import warnings
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Tuple, Type, Union
import requests
from gitlab import types
class _StdoutStream:

    def __call__(self, chunk: Any) -> None:
        print(chunk)