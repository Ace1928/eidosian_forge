from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import SSHKey
class SSHKeysPageResult(NamedTuple):
    ssh_keys: list[BoundSSHKey]
    meta: Meta | None