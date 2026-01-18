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
class ParallelCoordinatesPlotColumn(Base):
    metric: ParallelCoordinatesMetric
    display_name: Optional[str] = None
    inverted: Optional[bool] = None
    log: Optional[bool] = None
    _ref: Optional[internal.Ref] = Field(default_factory=lambda: None, init=False, repr=False)

    def to_model(self):
        obj = internal.Column(accessor=_metric_to_backend_pc(self.metric), display_name=self.display_name, inverted=self.inverted, log=self.log)
        obj.ref = self._ref
        return obj

    @classmethod
    def from_model(cls, model: internal.Column):
        obj = cls(metric=_metric_to_frontend_pc(model.accessor), display_name=model.display_name, inverted=model.inverted, log=model.log)
        obj._ref = model.ref
        return obj