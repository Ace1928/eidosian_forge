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
def _from_color_dict(d, runsets):
    d2 = {}
    for k, v in d.items():
        id, *backend_parts = k.split('-')
        if backend_parts:
            groups = []
            for part in backend_parts:
                key, value = part.rsplit(':', 1)
                kkey = _metric_to_frontend_panel_grid(key)
                group = RunsetGroupKey(kkey, value)
                groups.append(group)
            rs = _get_rs_by_id(runsets, id)
            rg = RunsetGroup(runset_name=rs.name, keys=groups)
            new_key = rg
        else:
            new_key = k
        d2[new_key] = v
    return d2