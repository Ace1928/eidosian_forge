import argparse
import collections
import logging
import sys
import curtsies
import curtsies.events
import curtsies.input
import curtsies.window
from . import args as bpargs, translations, inspection
from .config import Config
from .curtsiesfrontend import events
from .curtsiesfrontend.coderunner import SystemExitFromCodeRunner
from .curtsiesfrontend.interpreter import Interp
from .curtsiesfrontend.repl import BaseRepl
from .repl import extract_exit_value
from .translations import _
from typing import (
from ._typing_compat import Protocol
class SupportsEventGeneration(Protocol):

    def send(self, timeout: Optional[float]) -> Union[str, curtsies.events.Event, None]:
        ...

    def __iter__(self) -> 'SupportsEventGeneration':
        ...

    def __next__(self) -> Union[str, curtsies.events.Event, None]:
        ...