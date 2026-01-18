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
def _parse_option(cfg: configparser.RawConfigParser, cfg_opt_name: str, opt: str | None) -> list[str]:
    if opt is not None:
        return utils.parse_comma_separated_list(opt)
    else:
        for opt_name in (cfg_opt_name, cfg_opt_name.replace('_', '-')):
            val = cfg.get('flake8', opt_name, fallback=None)
            if val is not None:
                return utils.parse_comma_separated_list(val)
        else:
            return []