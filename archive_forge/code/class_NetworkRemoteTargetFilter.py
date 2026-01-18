from __future__ import annotations
import abc
import typing as t
from ...config import (
from ...util import (
from ...target import (
from ...host_configs import (
from ...host_profiles import (
class NetworkRemoteTargetFilter(RemoteTargetFilter[NetworkRemoteConfig]):
    """Target filter for remote network hosts."""