import argparse
import sys
from typing import (
import collections
from dataclasses import dataclass
import datetime
from enum import IntEnum
import logging
import math
import numbers
import numpy as np
import os
import pandas as pd
import textwrap
import time
from ray.air._internal.usage import AirEntrypoint
from ray.train import Checkpoint
from ray.tune.search.sample import Domain
from ray.tune.utils.log import Verbosity
import ray
from ray._private.dict import unflattened_lookup, flatten_dict
from ray._private.thirdparty.tabulate.tabulate import (
from ray.air.constants import TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.result import (
from ray.tune.experiment.trial import Trial
def _get_dict_as_table_data(data: Dict, include: Optional[Collection]=None, exclude: Optional[Collection]=None, upper_keys: Optional[Collection]=None):
    """Get ``data`` dict as table rows.

    If specified, excluded keys are removed. Excluded keys can either be
    fully specified (e.g. ``foo/bar/baz``) or specify a top-level dictionary
    (e.g. ``foo``), but no intermediate levels (e.g. ``foo/bar``). If this is
    needed, we can revisit the logic at a later point.

    The same is true for included keys. If a top-level key is included (e.g. ``foo``)
    then all sub keys will be included, too, except if they are excluded.

    If keys are both excluded and included, exclusion takes precedence. Thus, if
    ``foo`` is excluded but ``foo/bar`` is included, it won't show up in the output.
    """
    include = include or set()
    exclude = exclude or set()
    upper_keys = upper_keys or set()
    upper = []
    lower = []
    for key, value in sorted(data.items()):
        if key in exclude:
            continue
        for k, v in _render_table_item(str(key), value):
            if k in exclude:
                continue
            if include and key not in include and (k not in include):
                continue
            if key in upper_keys:
                upper.append([k, v])
            else:
                lower.append([k, v])
    if not upper:
        return lower
    elif not lower:
        return upper
    else:
        return upper + lower