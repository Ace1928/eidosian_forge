from __future__ import annotations
import configparser
import importlib.metadata
import inspect
import itertools
import logging
import sys
from typing import Any
from typing import Generator
from typing import Iterable
from typing import NamedTuple
from flake8 import utils
from flake8.defaults import VALID_CODE_PREFIX
from flake8.exceptions import ExecutionError
from flake8.exceptions import FailedToLoadPlugin
def _find_importlib_plugins() -> Generator[Plugin, None, None]:
    seen = set()
    for dist in importlib.metadata.distributions():
        eps = dist.entry_points
        if not any((ep.group in FLAKE8_GROUPS for ep in eps)):
            continue
        meta = dist.metadata
        if meta['name'] in seen:
            continue
        else:
            seen.add(meta['name'])
        if meta['name'] in BANNED_PLUGINS:
            LOG.warning('%s plugin is obsolete in flake8>=%s', meta['name'], BANNED_PLUGINS[meta['name']])
            continue
        elif meta['name'] == 'flake8':
            yield from _flake8_plugins(eps, meta['name'], meta['version'])
            continue
        for ep in eps:
            if ep.group in FLAKE8_GROUPS:
                yield Plugin(meta['name'], meta['version'], ep)