import datetime
import functools
import hashlib
import json
import logging
import optparse
import os.path
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.rich.console import Group
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import get_default_environment
from pip._internal.metadata.base import DistributionVersion
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.entrypoints import (
from pip._internal.utils.filesystem import adjacent_tmp_file, check_path_owner, replace
from pip._internal.utils.misc import ensure_dir
class SelfCheckState:

    def __init__(self, cache_dir: str) -> None:
        self._state: Dict[str, Any] = {}
        self._statefile_path = None
        if cache_dir:
            self._statefile_path = os.path.join(cache_dir, 'selfcheck', _get_statefile_name(self.key))
            try:
                with open(self._statefile_path, encoding='utf-8') as statefile:
                    self._state = json.load(statefile)
            except (OSError, ValueError, KeyError):
                pass

    @property
    def key(self) -> str:
        return sys.prefix

    def get(self, current_time: datetime.datetime) -> Optional[str]:
        """Check if we have a not-outdated version loaded already."""
        if not self._state:
            return None
        if 'last_check' not in self._state:
            return None
        if 'pypi_version' not in self._state:
            return None
        last_check = _convert_date(self._state['last_check'])
        time_since_last_check = current_time - last_check
        if time_since_last_check > _WEEK:
            return None
        return self._state['pypi_version']

    def set(self, pypi_version: str, current_time: datetime.datetime) -> None:
        if not self._statefile_path:
            return
        if not check_path_owner(os.path.dirname(self._statefile_path)):
            return
        ensure_dir(os.path.dirname(self._statefile_path))
        state = {'key': self.key, 'last_check': current_time.isoformat(), 'pypi_version': pypi_version}
        text = json.dumps(state, sort_keys=True, separators=(',', ':'))
        with adjacent_tmp_file(self._statefile_path) as f:
            f.write(text.encode())
        try:
            replace(f.name, self._statefile_path)
        except OSError:
            pass