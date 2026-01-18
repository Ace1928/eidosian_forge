import json
import logging
from datetime import datetime
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple
import click
import yaml
import ray._private.services as services
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.state import (
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException
from ray.util.annotations import PublicAPI
def _get_available_resources(excluded: Optional[List[StateResource]]=None) -> List[str]:
    """Return the available resources in a list of string

    Args:
        excluded: List of resources that should be excluded
    """
    return [e.value.replace('_', '-') for e in StateResource if excluded is None or e not in excluded]