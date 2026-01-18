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
def _load_spec_from_url(url, as_model=False):
    import json
    vs = _url_to_viewspec(url)
    spec = vs['spec']
    if as_model:
        return internal.Spec.model_validate_json(spec)
    return json.loads(spec)