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
def _metric_to_frontend_panel_grid(x: str):
    if x.startswith('config:') and '.value' in x:
        name = x.replace('config:', '').replace('.value', '')
        return Config(name)
    return _metric_to_frontend(x)