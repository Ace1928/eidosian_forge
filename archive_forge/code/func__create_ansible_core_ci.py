from __future__ import annotations
import os
import uuid
import configparser
import typing as t
from ....util import (
from ....config import (
from ....target import (
from ....core_ci import (
from ....host_configs import (
from . import (
def _create_ansible_core_ci(self) -> AnsibleCoreCI:
    """Return an AWS instance of AnsibleCoreCI."""
    return AnsibleCoreCI(self.args, CloudResource(platform='aws'))