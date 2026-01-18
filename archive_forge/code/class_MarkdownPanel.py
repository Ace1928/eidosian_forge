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
class MarkdownPanel(Panel):
    markdown: str = ''

    def to_model(self):
        obj = internal.MarkdownPanel(config=internal.MarkdownPanelConfig(value=self.markdown), layout=self.layout.to_model(), id=self.id)
        obj.ref = self._ref
        return obj

    @classmethod
    def from_model(cls, model: internal.ScatterPlot):
        obj = cls(markdown=model.config.value, layout=Layout.from_model(model.layout), id=model.id)
        obj._ref = model.ref
        return obj