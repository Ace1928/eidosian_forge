import pathlib
from os import path
from pprint import pformat
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from jinja2 import BaseLoader, FileSystemLoader, TemplateNotFound
from jinja2.environment import Environment
from jinja2.sandbox import SandboxedEnvironment
from jinja2.utils import open_if_exists
from sphinx.application import TemplateBridge
from sphinx.theming import Theme
from sphinx.util import logging
from sphinx.util.osutil import mtimes_of_files
def _slice_index(values: List, slices: int) -> Iterator[List]:
    seq = list(values)
    length = 0
    for value in values:
        length += 1 + len(value[1][1])
    items_per_slice = length // slices
    offset = 0
    for slice_number in range(slices):
        count = 0
        start = offset
        if slices == slice_number + 1:
            offset = len(seq)
        else:
            for value in values[offset:]:
                count += 1 + len(value[1][1])
                offset += 1
                if count >= items_per_slice:
                    break
        yield seq[start:offset]