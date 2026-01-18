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
@validator('runsets')
def _validate_list_not_empty(cls, v):
    if len(v) < 1:
        raise ValueError('must have at least one runset')
    return v