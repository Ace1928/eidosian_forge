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
class MediaBrowser(Panel):
    num_columns: Optional[int] = None
    media_keys: LList[str] = Field(default_factory=list)

    def to_model(self):
        obj = internal.MediaBrowser(config=internal.MediaBrowserConfig(column_count=self.num_columns, media_keys=self.media_keys), layout=self.layout.to_model(), id=self.id)
        obj.ref = self._ref
        return obj

    @classmethod
    def from_model(cls, model: internal.MediaBrowser):
        obj = cls(num_columns=model.config.column_count, media_keys=model.config.media_keys, layout=Layout.from_model(model.layout), id=model.id)
        obj._ref = model.ref
        return obj