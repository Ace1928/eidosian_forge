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
def _self_version_check_logic(*, state: SelfCheckState, current_time: datetime.datetime, local_version: DistributionVersion, get_remote_version: Callable[[], Optional[str]]) -> Optional[UpgradePrompt]:
    remote_version_str = state.get(current_time)
    if remote_version_str is None:
        remote_version_str = get_remote_version()
        if remote_version_str is None:
            logger.debug('No remote pip version found')
            return None
        state.set(remote_version_str, current_time)
    remote_version = parse_version(remote_version_str)
    logger.debug('Remote version of pip: %s', remote_version)
    logger.debug('Local version of pip:  %s', local_version)
    pip_installed_by_pip = was_installed_by_pip('pip')
    logger.debug('Was pip installed by pip? %s', pip_installed_by_pip)
    if not pip_installed_by_pip:
        return None
    local_version_is_older = local_version < remote_version and local_version.base_version != remote_version.base_version
    if local_version_is_older:
        return UpgradePrompt(old=str(local_version), new=remote_version_str)
    return None