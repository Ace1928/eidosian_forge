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
class InvalidPyProjectBuildRequires(DiagnosticPipError):
    """Raised when pyproject.toml an invalid `build-system.requires`."""
    reference = 'invalid-pyproject-build-system-requires'

    def __init__(self, *, package: str, reason: str) -> None:
        super().__init__(message=f'Can not process {escape(package)}', context=Text(f'This package has an invalid `build-system.requires` key in pyproject.toml.\n{reason}'), note_stmt='This is an issue with the package mentioned above, not pip.', hint_stmt=Text('See PEP 518 for the detailed specification.'))