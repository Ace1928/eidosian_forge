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
class VcsHashUnsupported(HashError):
    """A hash was provided for a version-control-system-based requirement, but
    we don't have a method for hashing those."""
    order = 0
    head = "Can't verify hashes for these requirements because we don't have a way to hash version control repositories:"