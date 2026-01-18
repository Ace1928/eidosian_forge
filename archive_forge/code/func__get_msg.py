import datetime
import itertools
import os
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from multiprocessing import Process
from typing import Any, Collection, Dict, NoReturn, Optional, Union, cast, overload
from .compat import Literal
from .tables import row, table
from .util import COLORS, ICONS, MESSAGES, can_render
from .util import color as _color
from .util import locale_escape, supports_ansi, wrap
def _get_msg(self, title: Any, text: Any, style: Optional[str]=None, show: bool=False, spaced: bool=False, exits: Optional[int]=None):
    if self.ignore_warnings and style == MESSAGES.WARN:
        show = False
    self._counts[style] += 1
    return self.text(title, text, color=style, icon=style, show=show, spaced=spaced, exits=exits)