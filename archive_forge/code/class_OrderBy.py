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
@dataclass(config=dataclass_config)
class OrderBy(Base):
    name: MetricType
    ascending: bool = False

    def to_model(self):
        return internal.SortKey(key=internal.SortKeyKey(name=_metric_to_backend(self.name)), ascending=self.ascending)

    @classmethod
    def from_model(cls, model: internal.SortKey):
        return cls(name=_metric_to_frontend(model.key.name), ascending=model.ascending)