from __future__ import annotations
import configparser
import logging
import os.path
from typing import Any
from flake8 import exceptions
from flake8.defaults import VALID_CODE_PREFIX
from flake8.options.manager import OptionManager
def _find_config_file(path: str) -> str | None:
    home = os.path.expanduser('~')
    try:
        home_stat = _stat_key(home) if home != '~' else None
    except OSError:
        home_stat = None
    dir_stat = _stat_key(path)
    while True:
        for candidate in ('setup.cfg', 'tox.ini', '.flake8'):
            cfg = configparser.RawConfigParser()
            cfg_path = os.path.join(path, candidate)
            try:
                cfg.read(cfg_path, encoding='UTF-8')
            except (UnicodeDecodeError, configparser.ParsingError) as e:
                LOG.warning('ignoring unparseable config %s: %s', cfg_path, e)
            else:
                if 'flake8' in cfg or 'flake8:local-plugins' in cfg:
                    return cfg_path
        new_path = os.path.dirname(path)
        new_dir_stat = _stat_key(new_path)
        if new_dir_stat == dir_stat or new_dir_stat == home_stat:
            break
        else:
            path = new_path
            dir_stat = new_dir_stat
    return None