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
class CustomChart(Panel):
    query: dict = Field(default_factory=dict)
    chart_name: str = Field(default_factory=dict)
    chart_fields: dict = Field(default_factory=dict)
    chart_strings: dict = Field(default_factory=dict)

    @classmethod
    def from_table(cls, table_name: str, chart_fields: dict=None, chart_strings: dict=None):
        return cls(query={'summaryTable': {'tableKey': table_name}}, chart_fields=chart_fields, chart_strings=chart_strings)

    def to_model(self):
        obj = internal.Vega2(config=internal.Vega2Config(), layout=self.layout.to_model())
        obj.ref = self._ref
        return obj

    @classmethod
    def from_model(cls, model: internal.ScatterPlot):
        obj = cls(layout=Layout.from_model(model.layout))
        obj._ref = model.ref
        return obj