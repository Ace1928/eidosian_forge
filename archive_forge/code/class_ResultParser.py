from __future__ import annotations
import itertools
import types
import warnings
from io import BytesIO
from typing import (
from urllib.parse import urlparse
from urllib.request import url2pathname
class ResultParser:

    def __init__(self):
        pass

    def parse(self, source: IO, **kwargs: Any) -> Result:
        """return a Result object"""
        pass