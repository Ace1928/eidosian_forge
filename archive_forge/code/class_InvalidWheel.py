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
class InvalidWheel(InstallationError):
    """Invalid (e.g. corrupt) wheel."""

    def __init__(self, location: str, name: str):
        self.location = location
        self.name = name

    def __str__(self) -> str:
        return f"Wheel '{self.name}' located at {self.location} is invalid."