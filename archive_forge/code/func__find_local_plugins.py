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
def _find_local_plugins(cfg: configparser.RawConfigParser) -> Generator[Plugin, None, None]:
    for plugin_type in ('extension', 'report'):
        group = f'flake8.{plugin_type}'
        for plugin_s in utils.parse_comma_separated_list(cfg.get('flake8:local-plugins', plugin_type, fallback='').strip(), regexp=utils.LOCAL_PLUGIN_LIST_RE):
            name, _, entry_str = plugin_s.partition('=')
            name, entry_str = (name.strip(), entry_str.strip())
            ep = importlib.metadata.EntryPoint(name, entry_str, group)
            yield Plugin('local', 'local', ep)