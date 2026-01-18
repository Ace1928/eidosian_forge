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
def _metric_to_backend(x: Optional[MetricType]):
    if x is None:
        return x
    if isinstance(x, str):
        return expr_parsing.to_backend_name(x)
    if isinstance(x, Metric):
        name = x.name
        return expr_parsing.to_backend_name(name)
    if isinstance(x, Config):
        name, *rest = x.name.split('.')
        rest = '.' + '.'.join(rest) if rest else ''
        return f'config.{name}.value{rest}'
    if isinstance(x, SummaryMetric):
        name = x.name
        return f'summary_metrics.{name}'
    raise Exception('Unexpected metric type')