from __future__ import annotations
import collections.abc as c
import contextlib
import datetime
import json
import os
import re
import shutil
import tempfile
import time
import typing as t
from ...encoding import (
from ...ansible_util import (
from ...executor import (
from ...python_requirements import (
from ...ci import (
from ...target import (
from ...config import (
from ...io import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...cache import (
from .cloud import (
from ...data import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...inventory import (
from .filters import (
from .coverage import (
class IntegrationEnvironment:
    """Details about the integration environment."""

    def __init__(self, test_dir: str, integration_dir: str, targets_dir: str, inventory_path: str, ansible_config: str, vars_file: str) -> None:
        self.test_dir = test_dir
        self.integration_dir = integration_dir
        self.targets_dir = targets_dir
        self.inventory_path = inventory_path
        self.ansible_config = ansible_config
        self.vars_file = vars_file