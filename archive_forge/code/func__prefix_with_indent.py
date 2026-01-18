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
def _prefix_with_indent(s: Union[Text, str], console: Console, *, prefix: str, indent: str) -> Text:
    if isinstance(s, Text):
        text = s
    else:
        text = console.render_str(s)
    return console.render_str(prefix, overflow='ignore') + console.render_str(f'\n{indent}', overflow='ignore').join(text.split(allow_blank=True))