from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
def _invalid_max_staleness_msg(max_staleness: Any) -> str:
    return 'maxStalenessSeconds must be a positive integer, not %s' % max_staleness