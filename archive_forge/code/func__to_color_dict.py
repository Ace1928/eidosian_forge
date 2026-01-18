import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
def _to_color_dict(custom_run_colors, runsets):
    d = {}
    for k, v in custom_run_colors.items():
        if isinstance(k, RunsetGroup):
            rs = _get_rs_by_name(runsets, k.runset_name)
            if not rs:
                continue
            id = rs._id
            kvs = []
            for keys in k.keys:
                kk = _metric_to_backend_panel_grid(keys.key)
                vv = keys.value
                kv = f'{kk}:{vv}'
                kvs.append(kv)
            linked = '-'.join(kvs)
            key = f'{id}-{linked}'
        else:
            key = k
        d[key] = v
    return d