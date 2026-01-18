import configparser
import contextlib
import locale
import logging
import pathlib
import re
import sys
from itertools import chain, groupby, repeat
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union
from pip._vendor.requests.models import Request, Response
from pip._vendor.rich.console import Console, ConsoleOptions, RenderResult
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text
class HashUnpinned(HashError):
    """A requirement had a hash specified but was not pinned to a specific
    version."""
    order = 3
    head = 'In --require-hashes mode, all requirements must have their versions pinned with ==. These do not:'