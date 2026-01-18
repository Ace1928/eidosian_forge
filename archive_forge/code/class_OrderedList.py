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
class OrderedList(List):
    items: LList[str] = Field(default_factory=lambda: [''])

    def to_model(self):
        children = [OrderedListItem(li).to_model() for li in self.items]
        return internal.List(children=children, ordered=True)