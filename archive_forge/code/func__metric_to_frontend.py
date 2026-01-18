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
def _metric_to_frontend(x: str):
    if x is None:
        return x
    if x.startswith('config.') and '.value' in x:
        name = x.replace('config.', '').replace('.value', '')
        return Config(name)
    if x.startswith('summary_metrics.'):
        name = x.replace('summary_metrics.', '')
        return SummaryMetric(name)
    name = expr_parsing.to_frontend_name(x)
    return Metric(name)